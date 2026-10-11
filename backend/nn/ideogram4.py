# Ideogram 4 transformer (single-stream DiT with MRoPE, QK-RMSNorm, AdaLN)
# Adapted from ideogram-oss/ideogram4 (modeling_ideogram4.py) for Forge 2.25
import math
import os

# Diagnostic logging is OFF by default. Set IDEOGRAM4_DEBUG=1 in the environment
# (webui-user.bat: set COMMANDLINE_ARGS=, plus `set IDEOGRAM4_DEBUG=1`) to re-enable
# the [Ideogram4]/[DBG] trace used while integrating the model.
_DEBUG = os.environ.get("IDEOGRAM4_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on", "debug")


def debug_log(*args, **kwargs):
    if _DEBUG:
        print(*args, **kwargs)

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.attention import attention_function


# --- Constants ---
SEQUENCE_PADDING_INDICATOR = -1
OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3
IMAGE_POSITION_OFFSET = 65536
PATCH_SIZE = 2  # transformer patch size: 128 = 32 ae_channels * 2 * 2
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)


class Ideogram4Config:
    emb_dim: int = 4608
    num_layers: int = 34
    num_heads: int = 18
    intermediate_size: int = 12288
    adanln_dim: int = 512
    in_channels: int = 128  # ae_channels(32) * patch_size^2(4)
    llm_features_dim: int = 4096 * len(QWEN3_VL_ACTIVATION_LAYERS)  # 53248
    rope_theta: int = 5_000_000
    mrope_section: tuple = (24, 20, 20)
    norm_eps: float = 1e-5

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: (B, num_heads, L, head_dim); cos/sin: (B, L, head_dim)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Ideogram4MRoPE(nn.Module):
    def __init__(self, head_dim, base, mrope_section):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = tuple(mrope_section)
        self.head_dim = head_dim

    @torch.no_grad()
    def forward(self, position_ids):
        # position_ids: (B, L, 3) int (t, h, w)
        batch_size, seq_len, _ = position_ids.shape
        pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)
        inv_freq = self.inv_freq.to(dtype=torch.float32)[None, None, :, None].expand(
            3, batch_size, -1, 1
        )
        freqs = inv_freq @ pos.unsqueeze(2)
        freqs = freqs.transpose(2, 3)  # (3, B, L, inv_freq_size)

        freqs_t = freqs[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[axis] * 3
            idx = torch.arange(offset, length, 3, device=freqs_t.device)
            freqs_t[..., idx] = freqs[axis][..., idx]

        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        return emb.cos(), emb.sin()


class Ideogram4RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Device guard: if a partial (lowvram) weight load left this parameter
        # on CPU while activations are on GPU, cast per-call instead of crashing.
        w = self.weight
        if w.device != x.device:
            w = w.to(x.device)
        return F.rms_norm(x, self.weight.shape, w, self.eps)


class Ideogram4Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, eps=1e-5):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.norm_q = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, segment_ids, cos, sin):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = self.norm_q(q)
        k = self.norm_k(k)

        # (B, L, H, D) -> (B, H, L, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # attention_function with skip_reshape=True returns (B, L, heads*head_dim)
        # already merged (see backend/attention.py: it permutes internally).
        # Do NOT transpose/reshape again - that was reordering heads/tokens
        # and produced garbage attention output (the root cause of "flat noise").
        out = attention_function(q, k, v, self.num_heads, skip_reshape=True)
        return self.o(out)


class Ideogram4MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Ideogram4TransformerBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, norm_eps, adanln_dim):
        super().__init__()
        self.attention = Ideogram4Attention(hidden_size, num_heads, eps=1e-5)
        self.feed_forward = Ideogram4MLP(hidden_size, intermediate_size)

        self.attention_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.attention_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)

        self.adaln_modulation = nn.Linear(adanln_dim, 4 * hidden_size, bias=True)

    def forward(self, x, segment_ids, cos, sin, adaln_input):
        mod = self.adaln_modulation(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa,
            segment_ids=segment_ids,
            cos=cos,
            sin=sin,
        )
        x = x + gate_msa * self.attention_norm2(attn_out)
        x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        return x


def _sinusoidal_embedding(t, dim, scale=1e4):
    t = t.to(torch.float32)
    half = dim // 2
    freq = math.log(scale) / (half - 1)
    freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)
    emb = t.unsqueeze(-1) * freq
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Ideogram4EmbedScalar(nn.Module):
    def __init__(self, dim, input_range):
        super().__init__()
        self.dim = dim
        self.range_min, self.range_max = input_range
        self.mlp_in = nn.Linear(dim, dim, bias=True)
        self.mlp_out = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        x = x.to(torch.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = _sinusoidal_embedding(scaled, self.dim)
        emb = emb.to(self.mlp_in.weight.dtype)
        emb = F.silu(self.mlp_in(emb))
        return self.mlp_out(emb)


class Ideogram4FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, adanln_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaln_modulation = nn.Linear(adanln_dim, hidden_size, bias=True)

    def forward(self, x, c):
        param_dtype = self.linear.weight.dtype
        scale = 1.0 + self.adaln_modulation(F.silu(c.to(param_dtype)))
        return self.linear(self.norm_final(x).to(param_dtype) * scale.to(param_dtype))


class Ideogram4Transformer(nn.Module):
    """Ideogram 4 flow-matching transformer (single-stream DiT)."""

    def __init__(self, config: Ideogram4Config = None):
        super().__init__()
        if config is None:
            config = Ideogram4Config()
        self.config = config

        head_dim = config.emb_dim // config.num_heads

        self.input_proj = nn.Linear(config.in_channels, config.emb_dim, bias=True)
        self.llm_cond_norm = Ideogram4RMSNorm(config.llm_features_dim, eps=1e-6)
        self.llm_cond_proj = nn.Linear(config.llm_features_dim, config.emb_dim, bias=True)
        self.t_embedding = Ideogram4EmbedScalar(config.emb_dim, input_range=(0.0, 1.0))
        self.adaln_proj = nn.Linear(config.emb_dim, config.adanln_dim, bias=True)

        self.embed_image_indicator = nn.Embedding(2, config.emb_dim)

        self.rotary_emb = Ideogram4MRoPE(
            head_dim=head_dim,
            base=config.rope_theta,
            mrope_section=config.mrope_section,
        )

        self.layers = nn.ModuleList([
            Ideogram4TransformerBlock(
                hidden_size=config.emb_dim,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_heads,
                norm_eps=config.norm_eps,
                adanln_dim=config.adanln_dim,
            )
            for _ in range(config.num_layers)
        ])

        self.final_layer = Ideogram4FinalLayer(
            hidden_size=config.emb_dim,
            out_channels=config.in_channels,
            adanln_dim=config.adanln_dim,
        )

        # Unconditional LoRA state (set by load_unconditional_lora).
        self._uncond_lora_active = False
        # Full unconditional transformer (official pipeline's separate model).
        # When set, the negative CFG pass runs on THIS module instead of self;
        # takes precedence over the LoRA imitation. Managed by the
        # ideogram4_studio extension; None = LoRA mode (default).
        self._uncond_model = None

    def set_unconditional_model(self, model):
        """Attach/detach a full unconditional transformer for the CFG
        negative branch. Pass None to fall back to the unconditional LoRA."""
        assert model is None or isinstance(model, Ideogram4Transformer), \
            "unconditional model must be an Ideogram4Transformer"
        self._uncond_model = model

    @property
    def device(self):
        return next(self.parameters()).device

    def load_unconditional_lora(self, lora_path, strength=1.0):
        """Load the unconditional LoRA and patch each targeted Linear module's forward
        so that, when self._uncond_lora_active is True (the negative CFG pass), the
        LoRA delta is added to the layer's output. The base (quantized) weights are
        left untouched — the delta is computed in float32 and added on top.
        """
        from safetensors import safe_open
        deltas = {}
        with safe_open(lora_path, framework="pt") as f:
            meta = f.metadata() or {}
            keys = list(f.keys())
            pairs = {}
            for k in keys:
                if k.endswith(".lora_A.weight"):
                    layer = k.replace(".lora_A.weight", "").replace("diffusion_model.", "")
                    pairs.setdefault(layer, {})["A"] = k
                elif k.endswith(".lora_B.weight"):
                    layer = k.replace(".lora_B.weight", "").replace("diffusion_model.", "")
                    pairs.setdefault(layer, {})["B"] = k
            for layer, parts in pairs.items():
                if "A" not in parts or "B" not in parts:
                    continue
                A = f.get_tensor(parts["A"]).float()
                B = f.get_tensor(parts["B"]).float()
                rank = A.shape[0]
                alpha = int(meta.get("alpha", rank))
                # Store the compact rank-r factors (not the full out_dim x in_dim
                # matrix) to keep VRAM small. The correction is computed on the fly.
                deltas[layer] = (A * (alpha / rank), B)

        # Patch each targeted module's forward to add the delta when active.
        # Factors stay on CPU here; they are lazily moved to x's device on first
        # use, so this works whether the model is on CPU or GPU at load time.
        patched = 0
        for layer, factors in deltas.items():
            mod = self
            for part in layer.split("."):
                mod = getattr(mod, part)
            orig_forward = mod.forward
            if getattr(mod, "_uncond_lora_patched", False):
                continue
            mod._uncond_lora_factors = factors
            mod._uncond_lora_patched = True

            def make_patched(orig, factors):
                A_scaled, B = factors
                def patched_forward(*args, **kwargs):
                    out = orig(*args, **kwargs)
                    if self._uncond_lora_active:
                        x = args[0] if args else kwargs.get("input")
                        dev = x.device
                        A = A_scaled.to(dev)
                        Bm = B.to(dev)
                        # corr = x @ (B @ A_scaled).T  = (x @ A_scaled.T) @ B.T
                        # Compute in bf16 to keep VRAM low (LoRA delta is small).
                        h = torch.matmul(x, A.to(x.dtype).t())
                        corr = torch.matmul(h, Bm.to(x.dtype).t()) * strength
                        out = out + corr
                    return out
                return patched_forward

            mod.forward = make_patched(orig_forward, factors)
            patched += 1

        self._uncond_lora_active = False
        import backend.memory_management as mm
        debug_log(f"[Ideogram4] unconditional LoRA loaded: {patched} layers patched, strength={strength}")
        return patched

    def forward(self, x, t, context=None, control=None, transformer_options=None, **kwargs):
        """Velocity prediction for Forge's KModel calling convention.

        Forge's KModel.apply_model calls:
            diffusion_model(xc, t, context=context, control=control, transformer_options=...)
        where xc is the image latent as (B, C, H, W) in packed 128-channel form,
        t is the sigma (flow-matching time in [0,1]), and context is a raw tensor
        (B, max_text_tokens, 53248) carrying the text encoding (like Flux2).

        ASYMMETRIC CFG (matches official pipeline_ideogram4.py):
        Forge's CFGDenoiser concatenates cond + uncond rows into ONE batch.
        The official model runs TWO separate transformers:
          * conditional pass   = full sequence [26 real text | 4096 image]
          * unconditional pass = IMAGE-ONLY [4096 image], zero features,
                                truncated positions/segments/indicators,
                                and its own (LoRA-adjusted) weights.
        We reproduce that in a single transformer by splitting this batch
        PER-ROW into two sub-forwards and stitching the outputs back.
        The 2.25 port instead ran one uniform 4122-token forward with ghost
        text slots in the negative pass and a batch-level LoRA flag that
        never fired on a mixed batch — that was the symmetric-CFG bug.
        """
        param_dtype = self.input_proj.weight.dtype

        # x is (B, 128, H, W) — packed latent from Forge
        B, C, H, W = x.shape
        grid_h, grid_w = H, W
        num_image_tokens = grid_h * grid_w
        debug_log(f"[DBG fwd] x.shape={tuple(x.shape)} context={tuple(context.shape)} "
              f"num_image_tokens={num_image_tokens}", flush=True)

        # Context: (B, max_text_tokens, 53248). Negative rows are zero
        # (the engine feeds zeros for the unconditional branch).
        llm_features = context.to(param_dtype)  # (B, max_text_tokens, D)
        max_text_tokens = llm_features.shape[1]
        D = llm_features.shape[2]

        # Per-row negative detection on the RAW (unpadded) text features.
        # Reduce over BOTH token and feature dims -> one bool per batch row.
        neg_mask = (llm_features.float().abs().amax(dim=(1, 2)) == 0.0)  # (B,)
        debug_log(f"[DBG prologue] llm_features={tuple(llm_features.shape)} neg_mask={neg_mask.tolist()}", flush=True)

        # Reshape latent to tokens with patchify reorder, matching ComfyUI/official.
        # Forge latent is (B, 128, H, W) with the 128 dim ordered [c, pi, pj].
        # The transformer expects tokens ordered [pi, pj, c] (see official _decode).
        ae_channels = C // (PATCH_SIZE * PATCH_SIZE)
        x_tokens = x.view(B, ae_channels, PATCH_SIZE, PATCH_SIZE, H, W)
        x_tokens = x_tokens.permute(0, 4, 5, 2, 3, 1).reshape(B, H * W, C)
        x_tokens = x_tokens.to(param_dtype)

        # t is the sigma from Forge (flow-matching time in [0,1])
        t = t.to(param_dtype)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(B)
        elif t.dim() == 1:
            pass  # already (B,)

        # Image positions: (0, h_idx, w_idx) + IMAGE_POSITION_OFFSET
        # Must be on x.device: the negative branch feeds these straight into
        # MRoPE, whose inv_freq buffer lives on the model device (CPU arange
        # there causes a device-mismatch crash in the bmm).
        h_idx = torch.arange(grid_h, device=x.device).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
        w_idx = torch.arange(grid_w, device=x.device).view(1, -1).expand(grid_h, grid_w).reshape(-1)
        t_idx = torch.zeros_like(h_idx)
        image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET  # (L_img, 3)

        out_full = torch.zeros(B, num_image_tokens, C, dtype=param_dtype, device=x.device)

        pos_idx = (~neg_mask).nonzero(as_tuple=True)[0]
        neg_idx = neg_mask.nonzero(as_tuple=True)[0]

        # ---- Positive (conditional) pass: [real text | image] ----
        if len(pos_idx) > 0:
            x_pos = x_tokens.index_select(0, pos_idx)          # (P, L_img, 128)
            feat_pos = llm_features.index_select(0, pos_idx)   # (P, T, D)

            text_z_padding = torch.zeros(len(pos_idx), max_text_tokens, C, dtype=x_pos.dtype, device=x_pos.device)
            full_x = torch.cat([text_z_padding, x_pos], dim=1)

            L = max_text_tokens + num_image_tokens
            debug_log(f"[DBG pos] max_text_tokens={max_text_tokens} num_image_tokens={num_image_tokens} L={L} feat_pos={tuple(feat_pos.shape)}", flush=True)
            position_ids = torch.zeros(len(pos_idx), L, 3, dtype=torch.long, device=x.device)
            text_pos = torch.arange(max_text_tokens, device=x.device)
            text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
            position_ids[:, :max_text_tokens] = text_pos_3d
            position_ids[:, max_text_tokens:] = image_pos

            indicator = torch.full((len(pos_idx), L), LLM_TOKEN_INDICATOR, dtype=torch.long, device=x.device)
            indicator[:, max_text_tokens:] = OUTPUT_IMAGE_INDICATOR

            segment_ids = torch.ones(len(pos_idx), L, dtype=torch.long, device=x.device)

            llm_pad = feat_pos
            if llm_pad.shape[1] < L:
                pad = torch.zeros(len(pos_idx), L - llm_pad.shape[1], D, dtype=llm_pad.dtype, device=llm_pad.device)
                llm_pad = torch.cat([llm_pad, pad], dim=1)

            self._uncond_lora_active = False
            out = self._forward_packed(
                llm_features=llm_pad,
                x=full_x,
                t=t.index_select(0, pos_idx),
                position_ids=position_ids,
                segment_ids=segment_ids,
                indicator=indicator,
            )
            out_full.index_copy_(0, pos_idx, out[:, max_text_tokens:].to(param_dtype))

        # ---- Negative (unconditional) pass: image-only, zero features ----
        if len(neg_idx) > 0:
            x_neg = x_tokens.index_select(0, neg_idx)          # (N, L_img, 128)

            L = num_image_tokens  # no text tokens at all
            position_ids = image_pos.unsqueeze(0).expand(len(neg_idx), -1, -1)

            indicator = torch.full((len(neg_idx), L), OUTPUT_IMAGE_INDICATOR, dtype=torch.long, device=x.device)
            segment_ids = torch.ones(len(neg_idx), L, dtype=torch.long, device=x.device)

            llm_neg = torch.zeros(len(neg_idx), L, D, dtype=param_dtype, device=x.device)

            # Official unconditional branch = a SEPARATE unconditional
            # transformer. When the ideogram4_studio extension attached one
            # (_uncond_model), use it directly. Otherwise this single-
            # transformer design falls back to the unconditional LoRA delta.
            uncond_model = self._uncond_model
            if uncond_model is not None:
                if next(uncond_model.parameters()).device != x.device:
                    uncond_model.to(x.device)
                out = uncond_model._forward_packed(
                    llm_features=llm_neg,
                    x=x_neg,
                    t=t.index_select(0, neg_idx),
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    indicator=indicator,
                )
            else:
                self._uncond_lora_active = True
                out = self._forward_packed(
                    llm_features=llm_neg,
                    x=x_neg,
                    t=t.index_select(0, neg_idx),
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    indicator=indicator,
                )
                self._uncond_lora_active = False
            out_full.index_copy_(0, neg_idx, out.to(param_dtype))

        try:
            import backend.memory_management as mm
            of = out_full.float()
            debug_log(
                f"[Ideogram4 forward] B={B} pos={len(pos_idx)} neg={len(neg_idx)} "
                f"t={t[0].item():.4f} n_img_tokens={num_image_tokens} "
                f"vel_std={of.std().item():.4f} vel_absmax={of.abs().max().item():.4f} "
                f"vel_mean={of.mean().item():.4f} finite={bool(torch.isfinite(of).all())} "
                f"x_std={x.float().std().item():.4f}"
            )
        except Exception:
            pass

        # Negate velocity: Ideogram4 points image→noise (increasing t),
        # Forge's Euler step expects noise→image (decreasing sigma)
        out_img = -out_full

        # Unpatchify back to Forge latent order [c, pi, pj] (inverse of the
        # patchify above; matches ComfyUI _tokens_to_img / official _decode).
        out_img = out_img.view(B, H, W, PATCH_SIZE, PATCH_SIZE, ae_channels)
        out_img = out_img.permute(0, 5, 3, 4, 1, 2).reshape(B, C, H, W)
        return out_img

    def _forward_packed(self, *, llm_features, x, t, position_ids, segment_ids, indicator):
        """Core transformer forward on packed token sequence."""
        param_dtype = self.input_proj.weight.dtype
        debug_log(f"[DBG packed] llm={tuple(llm_features.shape)} x={tuple(x.shape)} "
              f"pos_ids={tuple(position_ids.shape)} ind={tuple(indicator.shape)}", flush=True)
        x = x.to(param_dtype)
        t = t.to(param_dtype)
        llm_features = llm_features.to(param_dtype)

        indicator = indicator.to(torch.long)
        llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
        output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x.dtype).unsqueeze(-1)

        llm_features = llm_features * llm_token_mask
        x = x * output_image_mask

        x = self.input_proj(x) * output_image_mask

        t_cond = self.t_embedding(t)
        if t.dim() == 1:
            t_cond = t_cond.unsqueeze(1)
        adaln_input = F.silu(self.adaln_proj(t_cond.to(param_dtype)))

        llm_features = self.llm_cond_norm(llm_features)
        llm_features = self.llm_cond_proj(llm_features) * llm_token_mask

        h = x + llm_features

        image_indicator_embedding = self.embed_image_indicator(
            (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long)
        )
        h = h + image_indicator_embedding

        cos, sin = self.rotary_emb(position_ids)
        cos = cos.to(h.dtype)
        sin = sin.to(h.dtype)

        for layer in self.layers:
            h = layer(h, segment_ids=segment_ids, cos=cos, sin=sin, adaln_input=adaln_input)

        out = self.final_layer(h, c=adaln_input)
        return out.to(torch.float32)
