# Ideogram 4 diffusion engine for Forge 2.29.2
# Adapted from VERIFIED 2.25 engine (e2e-tested with LoRA, produces real images).
# Key changes vs 2.25:
#   * Uses 2.29.2 imports (memory_management.cast_to, is_flux2 flag, etc.)
#   * CRITICAL FIX: Per-row asymmetric CFG in _forward_packed.
#     Forge's CFGDenoiser concatenates cond+uncond into ONE batch.
#     The 2.25 port detected negative pass at batch level (llm_features.all_zero),
#     which failed when mixed rows reached the transformer.
#     Official uses TWO separate transformers; we implement per-row split in ONE.
#   * Logit-normal schedule lives in the predictor (same as 2.25).
#   * CPU-only Qwen3VLTextModel encoding (abliterated encoder).
#   * Unconditional LoRA (applied only on negative rows via per-row flag).

import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from huggingface_guess import model_list

from backend import memory_management
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.modules.k_prediction import AbstractPrediction
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

# --- Diagnostic logging (OFF by default; enable with IDEOGRAM4_DEBUG=1) ---
_DEBUG = os.environ.get("IDEOGRAM4_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on", "debug")


def debug_log(*args, **kwargs):
    if _DEBUG:
        print(*args, **kwargs)


# --- Constants from official ideogram-oss/ideogram4 ---
SEQUENCE_PADDING_INDICATOR = -1
OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3
IMAGE_POSITION_OFFSET = 65536
# Qwen3-VL-8B activation layers tapped by official pipeline (post-layer outputs)
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
# Forge's intermediate_output captures PRE-layer output = official POST-layer (i-1)
FORGE_TAP_LAYERS = tuple(l + 1 for l in QWEN3_VL_ACTIVATION_LAYERS)
PATCH_SIZE = 2
AE_SCALE_FACTOR = 8

# --- Logit-normal schedule (official ideogram4/scheduler.py) ---
@dataclass(frozen=True)
class LogitNormalSchedule:
    mean: float
    std: float = 1.0
    logsnr_min: float = -15.0
    logsnr_max: float = 18.0

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(torch.float64)
        z = torch.special.ndtri(t)
        y = self.mean + self.std * z
        t_ = torch.special.expit(y)
        t_ = 1.0 - t_
        t_min = 1.0 / (1 + math.exp(0.5 * self.logsnr_max))
        t_max = 1.0 / (1 + math.exp(0.5 * self.logsnr_min))
        return t_.clamp(t_min, t_max).to(torch.float32)

def get_schedule_for_resolution(image_resolution, known_resolution=(512, 512), known_mean=1.0, std=1.0):
    num_pixels = image_resolution[0] * image_resolution[1]
    known_pixels = known_resolution[0] * known_resolution[1]
    mean = known_mean + 0.5 * math.log(num_pixels / known_pixels)
    return LogitNormalSchedule(mean=mean, std=std)

# --- Custom k_predictor for Ideogram4 flow-matching ---
class PredictionIdeogram4(AbstractPrediction):
    """Flow-matching predictor with a logit-normal time schedule.

    prediction_type='const': model input is raw latent (no sigma scaling),
    model output is used directly as velocity.

    `sigma()` maps the standard 1..N ramp (Forge's "Simple" schedule) onto
    the logit-normal curve. `timestep()` converts Forge sigma (1=noise) to
    official model time t (0=noise): t = 1 - sigma.
    """

    def __init__(self, model_config):
        super().__init__(sigma_data=None, prediction_type="const")
        ss = model_config.sampling_settings
        self._mu = ss.get("mu", 0.0)
        self._std = ss.get("std", 1.75)
        self._num_steps = ss.get("num_steps", 20)
        self._height = 1024
        self._width = 1024
        self._schedule = get_schedule_for_resolution((self._height, self._width), known_mean=self._mu, std=self._std)
        self.register_buffer("sigmas", self._build_sigmas())
        self._refresh_polish_threshold()

    def _refresh_polish_threshold(self):
        # Official guidance_schedule (sampler_configs.py / Comfy CFGOverride
        # [3, 0.9, 1]): ALL steps at gw except the last ~10%, which polish at
        # gw=3. Our sigmas buffer is ascending (index 0 = image, last = noise);
        # the "Simple" scheduler samples it uniformly by index, so regardless of
        # the user's step count the last 10% of steps are exactly the steps
        # with sigma <= sigmas[k], k = max(1, round(0.1 * len(sigmas))).
        n = len(self.sigmas)
        k = max(1, int(round(0.1 * n)))
        self._polish_sigma = float(self.sigmas[min(k, n - 1)].item())

    def set_resolution(self, height, width, num_steps=None, mu=None, std=None):
        if num_steps is not None:
            self._num_steps = num_steps
        if mu is not None:
            self._mu = mu
        if std is not None:
            self._std = std
        self._height, self._width = height, width
        self._schedule = get_schedule_for_resolution((height, width), known_mean=self._mu, std=self._std)
        self.register_buffer("sigmas", self._build_sigmas())
        self._refresh_polish_threshold()
        memory_management.logger.debug(
            f"[Ideogram4] schedule: {self._num_steps} steps, mu={self._mu}, std={self._std}, "
            f"sigmas[0]={self.sigmas[0]:.4f}, sigmas[-1]={self.sigmas[-1]:.4f}"
        )

    def _build_sigmas(self):
        # Forge stores sigmas small->large (image->noise), matching Flux2.
        # Simple scheduler reverses into decreasing noise->image order.
        # Forge sigma = 1 - t_official, where t_official = schedule(ramp).
        ramp = torch.linspace(0.0, 1.0, self._num_steps + 1, dtype=torch.float32)
        return 1.0 - self._schedule(ramp)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        # Forge sigma: 1=noise, 0=image. Official t: 0=noise, 1=image.
        t = 1.0 - sigma
        try:
            if torch.is_tensor(sigma) and sigma.numel() == 1:
                debug_log(
                    f"[DBG step] sigma={sigma.item():.6f} t_model={t.item():.6f}"
                )
        except Exception:
            pass
        return t

    def sigma(self, timestep):
        return self._schedule(timestep)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return self._schedule(torch.tensor(1.0 - percent, dtype=torch.float32)).item()

# --- Prompt normalization: raw text -> official Ideogram JSON caption ---
# Ideogram 4 is trained on structured JSON captions (see official caption_verifier
# and ComfyUI BuildJsonPromptIdeogram node). Raw text or truncated/broken JSON
# makes the model emit its learned "refusal card" (a rendered text placeholder).
# Rules (from official CaptionVerifier):
#   * root keys: high_level_description, style_description, compositional_deconstruction
#   * compositional_deconstruction MUST exist and must contain a background element
#     covering the full canvas (bbox [0,0,1000,1000])
#   * style_description needs exactly one of photo/art_style
# Helpers: the WebUI clipboard/paste path can deliver a TRUNCATED or
# double-escaped JSON prompt ("...color_palette":[" with no close). The old
# wrapper used such garbled text verbatim as the description (306 tokens of
# escaped JSON gibberish) -> out-of-distribution input -> model renders its
# learned refusal card. So: first try to REPAIR the JSON (close truncation),
# then try to EXTRACT the human description from the garble.
def _repair_truncated_json(s):
    """Close a truncated JSON document by appending bracket/quote tokens."""
    import itertools
    t = s.strip()
    if not t.startswith("{"):
        return None
    for n in range(0, 5):
        for combo in itertools.product(['"', ']', '}'], repeat=n):
            try:
                d = json.loads(t + "".join(combo))
            except Exception:
                continue
            if isinstance(d, dict):
                return d
    return None

def _desc_from_garbled(s):
    """Extract the human-readable description from garbled/escaped JSON text."""
    import re
    cleaned = s.replace('\\"', '"')
    m = re.search(r'"high_level_description"\s*:\s*"([^"]+)"', cleaned)
    if m:
        return m.group(1).strip()
    frags = re.findall(r"[A-Za-z][A-Za-z0-9 ,.'\-]{20,}", cleaned)
    if frags:
        return max(frags, key=len).strip()
    return ""

def normalize_prompt_to_json(prompt):
    """Accept dict or str; return a valid Ideogram JSON caption string.

    - valid caption dict/JSON -> structure-checked, content kept verbatim
    - truncated/double-escaped JSON -> repaired (close brackets) or, failing
      that, the high_level_description text is extracted from the garble
    - raw text -> wrapped into a minimal-but-valid caption (ComfyUI
      BuildJsonPromptIdeogram pattern: description becomes high_level_description
      AND a full-canvas background element)
    """
    caption = None
    if isinstance(prompt, dict):
        caption = prompt
    elif isinstance(prompt, str):
        s = prompt.strip()
        if s.startswith("{"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    caption = parsed
            except json.JSONDecodeError:
                caption = _repair_truncated_json(s)
                if caption is not None:
                    debug_log("[Ideogram4] truncated JSON prompt repaired")
            # double-escaped wrapper: {"high_level_description": "<escaped JSON>"}
            if isinstance(caption, dict):
                hld = caption.get("high_level_description")
                if isinstance(hld, str) and hld.strip().startswith("{"):
                    inner = _repair_truncated_json(hld)
                    if isinstance(inner, dict):
                        caption = inner
                        debug_log("[Ideogram4] unwrapped double-escaped inner caption")
                    else:
                        extracted = _desc_from_garbled(hld)
                        if extracted:
                            caption["high_level_description"] = extracted
                            debug_log("[Ideogram4] extracted description from garbled inner JSON")
    if caption is None:
        raw = prompt if isinstance(prompt, str) else str(prompt)
        raw = raw.strip()
        if raw.startswith("{") or raw.startswith("["):
            extracted = _desc_from_garbled(raw)
            if extracted:
                debug_log("[Ideogram4] garbled JSON prompt: extracted description only")
                raw = extracted
        if not raw:
            raw = "A simple scene."
        caption = {
            "high_level_description": raw,
            "style_description": {
                "aesthetics": "natural, coherent",
                "lighting": "soft natural lighting",
                "medium": "photograph",
                "photo": "high resolution",
            },
            "compositional_deconstruction": {
                "background": raw,
                "elements": [
                    {"type": "obj", "bbox": [0, 0, 1000, 1000], "desc": raw},
                ],
            },
        }
        debug_log("[Ideogram4] raw prompt wrapped into JSON caption")
        return json.dumps(caption, ensure_ascii=False)

    # We have a caption dict: enforce the mandatory structure without touching
    # the user's content.
    cd = caption.get("compositional_deconstruction")
    if not isinstance(cd, dict):
        hld = caption.get("high_level_description", "A simple scene.")
        if not isinstance(hld, str) or not hld.strip():
            hld = "A simple scene."
        caption["compositional_deconstruction"] = {
            "background": hld,
            "elements": [{"type": "obj", "bbox": [0, 0, 1000, 1000], "desc": hld}],
        }
        debug_log("[Ideogram4] caption missing compositional_deconstruction: added full-canvas background element")
    else:
        elements = cd.get("elements")
        if not isinstance(elements, list) or not elements:
            hld = caption.get("high_level_description", "A simple scene.")
            if not isinstance(hld, str) or not hld.strip():
                hld = str(cd.get("background") or "A simple scene.")
            cd["elements"] = [{"type": "obj", "bbox": [0, 0, 1000, 1000], "desc": hld}]
            debug_log("[Ideogram4] caption had no elements: added full-canvas background element")
        else:
            # ensure a full-canvas background element exists
            def _full_bbox(el):
                bb = el.get("bbox") if isinstance(el, dict) else None
                return isinstance(bb, list) and len(bb) == 4 and bb[0] <= 0 and bb[1] <= 0 and bb[2] >= 1000 and bb[3] >= 1000
            if not any(_full_bbox(el) for el in elements):
                bg = cd.get("background") or caption.get("high_level_description") or "the scene background"
                if not isinstance(bg, str) or not bg.strip():
                    bg = "the scene background"
                elements.insert(0, {"type": "obj", "bbox": [0, 0, 1000, 1000], "desc": bg})
                debug_log("[Ideogram4] inserted full-canvas background element (verifier requires it)")

    sd = caption.get("style_description")
    if isinstance(sd, dict) and ("photo" in sd) == ("art_style" in sd):
        # both or neither: keep whichever exists, else add a neutral photo key
        if "art_style" in sd:
            sd.pop("photo", None)
        else:
            sd["photo"] = sd.pop("photo", "high resolution") or "high resolution"
        debug_log("[Ideogram4] style_description: normalized to a single photo/art_style key")

    return json.dumps(caption, ensure_ascii=False)

# --- Text encoding (Qwen3-VL-8B; weights come from the TE file SELECTED in Forge) ---
_QWEN3VL_MODEL = None
_QWEN3VL_TOKENIZER = None
_QWEN3VL_PATCHER = None
_QWEN3VL_SOURCE = None  # weight file we built the current model from (for change detection)
_SELECTED_TE_FILE = None  # set by engine init from loader routing (guess.te_source_files)
_HF_TE_REPO = "noelthekke/ideogram4-text-encoder-abliterated"
# Tokenizer/config for the Qwen3 text tower is FIXED (verified byte-identical
# between the official Ideogram release and the abliterated TE repo: same model,
# vocab 151643, same added_tokens). It ships inside Forge under
# backend/huggingface/Ideogram4/ — "text_encoder_merged" is config.json +
# tokenizer files + the official chat template in one dir, sufficient for
# AutoConfig/AutoTokenizer. The Qwen3 tokenizer does NOT vary by TE variant
# (abliterated/vanilla/int8 tokenize identically), so there is no reason to
# take it from wherever a TE weight file happens to live.
_BUILTIN_CFG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "huggingface", "Ideogram4", "text_encoder_merged")

def _te_weights_to_bf16(sd):
    """Normalize a Qwen3-VL text-tower state dict for the HF model.

    Strips lm_head/visual/comfy-quant plumbing, dequantizes fp8 (and fp8_scaled
    with weight_scale tensors) to bf16. int8/convrot quants are not exactly
    invertible here and raise.
    """
    out = {}
    scales = {}
    for k, v in sd.items():
        if k.endswith(".weight_scale") or k.endswith(".weight_absmax"):
            scales[k.rsplit(".", 1)[0]] = v
            continue
        if k.endswith(".comfy_quant") or k.endswith(".permute"):
            continue
        if k.startswith("lm_head.") or "visual" in k:
            continue
        out[k] = v
    for k, v in out.items():
        if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            w = v.to(torch.float32)
            s = scales.get(k)
            if s is not None:
                w = w * s.to(torch.float32).reshape(-1, *([1] * (w.dim() - 1)))
            out[k] = w.to(torch.bfloat16)
        elif v.dtype in (torch.int8, torch.uint8):
            raise NotImplementedError(
                f"int8/convrot TE ({k}) cannot be dequantized exactly here; "
                "select an fp8 or bf16 text encoder")
    return out

def _ideogram_te_config_dir():
    """Config/tokenizer dir for the Qwen3 text tower.

    Fixed tokenizer: always the built-in copy shipped with this Forge install
    (backend/huggingface/Ideogram4/text_encoder_merged). No external paths.
    HF download remains only as last-resort repair if the built-in copy was
    removed from the installation.
    """
    if os.path.isfile(os.path.join(_BUILTIN_CFG_DIR, "config.json")) and os.path.isfile(os.path.join(_BUILTIN_CFG_DIR, "tokenizer.json")):
        return _BUILTIN_CFG_DIR
    debug_log(
        f"[Ideogram4] built-in TE config missing at {_BUILTIN_CFG_DIR}; downloading tokenizer/config from HF ({_HF_TE_REPO})")
    from huggingface_hub import snapshot_download
    return snapshot_download(
        repo_id=_HF_TE_REPO,
        allow_patterns=["config.json", "tokenizer*", "vocab.json", "merges.txt", "chat_template.jinja",
                       "added_tokens.json", "special_tokens_map.json"],
    )

def _load_selected_encoder():
    """Build the HF Qwen3VLTextModel from the TE file selected in Forge UI.

    Weights come from the selected file only; if none is selected we download
    the reference abliterated TE from HF. Config/tokenizer always come from the
    built-in copy (fixed tokenizer, see _BUILTIN_CFG_DIR). The model is
    registered with Forge's ModelPatcher so VRAM placement is decided by the
    memory manager (GPU when it fits).
    """
    global _QWEN3VL_MODEL, _QWEN3VL_TOKENIZER, _QWEN3VL_PATCHER, _QWEN3VL_SOURCE, _SELECTED_TE_FILE
    from transformers import AutoTokenizer, AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    from safetensors import safe_open

    weight_file = _SELECTED_TE_FILE
    if weight_file is None or not os.path.isfile(weight_file):
        debug_log(
            "[Ideogram4] no TE selected in UI; downloading reference abliterated TE from HF")
        from huggingface_hub import hf_hub_download
        weight_file = hf_hub_download(repo_id=_HF_TE_REPO, filename="qwen3vl_text_encoder_abliterated.safetensors")
    if _QWEN3VL_MODEL is not None and _QWEN3VL_SOURCE == weight_file:
        return
    if _QWEN3VL_MODEL is not None:
        debug_log(f"[Ideogram4] selected TE changed -> reloading encoder from {weight_file}")

    debug_log(f"[Ideogram4] Loading Qwen3VLTextModel weights from selected TE: {weight_file}")
    # Config/tokenizer are architecture-fixed for Ideogram4 (Qwen3-VL text
    # tower); always use the built-in copy regardless of where the weight
    # file came from.
    cfg_dir = _ideogram_te_config_dir()
    _QWEN3VL_TOKENIZER = AutoTokenizer.from_pretrained(cfg_dir)
    cfg = AutoConfig.from_pretrained(cfg_dir)
    # Enforce Qwen3-VL-8B rope settings regardless of the source config
    try:
        if getattr(cfg, "rope_parameters", None) is not None:
            cfg.rope_parameters["rope_theta"] = 5000000
            cfg.rope_scaling = dict(cfg.rope_parameters)
        if getattr(cfg, "rope_scaling", None) is None:
            cfg.rope_scaling = {"rope_type": "default", "rope_theta": 5000000}
        if getattr(cfg, "rope_theta", None) == 1000000.0:
            cfg.rope_theta = 5000000.0
    except Exception:
        pass
    # Build the architecture directly in bf16. The default-dtype constructor
    # allocates the whole 8B model in fp32 (~32 GB) before we overwrite the
    # weights — that full-RAM spike on top of every other resident model is
    # what pushed the system into swap. Buffers (rope inv_freq) stay fp32
    # regardless (created under float32 autocast), so numerics are unchanged.
    _prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        _QWEN3VL_MODEL = Qwen3VLTextModel(cfg)
    finally:
        torch.set_default_dtype(_prev_dtype)
    # Fill parameters one tensor at a time (lazy safetensors access, in-place).
    # No bulk dicts: peak RAM ~= model size (~15 GB), not 2-3x it.
    params = dict(_QWEN3VL_MODEL.named_parameters())
    with safe_open(weight_file, framework="pt") as f:
        file_keys = set(f.keys())
        for name, p in params.items():
            src = None
            for cand in (name, f"model.{name}"):
                if cand in file_keys:
                    src = cand
                    break
            if src is None:
                continue  # buffers etc.: keep constructor value
            t = f.get_tensor(src)
            if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                w = t.to(torch.float32)
                scale = None
                for sk in (f"{src}.weight_scale", f"{src}.weight_absmax"):
                    if sk in file_keys:
                        scale = f.get_tensor(sk).to(torch.float32)
                        break
                if scale is not None:
                    w = w * scale.reshape(-1, *([1] * (w.dim() - 1)))
                t = w.to(torch.bfloat16)
            elif t.dtype in (torch.int8, torch.uint8):
                # two int-width formats:
                #  * comfy int8_tensorwise / convrot: int8 weight + weight_scale
                #    (+ optional comfy_quant descriptor). Embedding layers (no
                #    comfy_quant) are plain per-row int8 -> direct dequant.
                #  * bitsandbytes nf4/fp4 (official ideogram-4-nf4 TE format):
                #    packed uint8 + sibling stat tensors ("...absmax",
                #    "...quant_map", "...quant_state.bitsandbytes__nf4").
                prefix = src[: -len(".weight")] if src.endswith(".weight") else src
                stats = {
                    k[len(prefix) + 1:]: f.get_tensor(k)
                    for k in file_keys
                    if k.startswith(prefix + ".") and k != src
                }
                conf_key, scale_key = f"{prefix}.comfy_quant", f"{prefix}.weight_scale"
                if t.dtype == torch.int8 and "comfy_quant" not in stats and "weight_scale" in stats:
                    # embedding table: per-row int8 scales, no hadamard/convrot
                    scale = f.get_tensor(scale_key).to(torch.float32)
                    t = (t.to(torch.float32) * scale.reshape(-1, *([1] * (t.dim() - 1)))).to(torch.bfloat16)
                elif "quant_state." not in "".join(stats.keys()):
                    if t.dtype != torch.int8 or "comfy_quant" not in stats or "weight_scale" not in stats:
                        raise NotImplementedError(
                            f"unsupported TE quantization for {src}; "
                            "select an fp8, int8, nf4 or bf16 text encoder")
                    import json as _json
                    from comfy_kitchen.tensor import QuantizedTensor, get_layout_class
                    conf = _json.loads(f.get_tensor(conf_key).numpy().tobytes())
                    if conf.get("format") != "int8_tensorwise":
                        raise NotImplementedError(
                            f"TE quant format '{conf.get('format')}' ({src}) not supported here")
                    layout_cls = get_layout_class("TensorWiseINT8Layout")
                    dev = memory_management.get_torch_device()
                    params = layout_cls.Params(
                        scale=f.get_tensor(scale_key).to(torch.float32).to(dev),
                        convrot=bool(conf.get("convrot", False)),
                        convrot_groupsize=int(conf.get("convrot_groupsize", 256)),
                        orig_dtype=torch.bfloat16, orig_shape=list(t.shape),
                    )
                    qt = QuantizedTensor(t.to(dev), "TensorWiseINT8Layout", params)
                    t = qt.dequantize()
                    del qt, params
                else:
                    # bitsandbytes nf4/fp4 (official ideogram-4-nf4 TE format):
                    # packed uint8 + sibling stat tensors ("...absmax",
                    # "...quant_map", "...quant_state.bitsandbytes__nf4").
                    # Dequantize through bnb CUDA kernels (the official
                    # pipeline also requires CUDA for bnb weights).
                    if not memory_management.bnb_enabled():
                        raise NotImplementedError('bitsandbytes is not installed; nf4 TE needs "bitsandbytes"')
                    import bitsandbytes as bnb
                    from backend.operations_bnb import functional_dequantize_4bit
                    qw = bnb.nn.Params4bit.from_prequantized(
                        data=t, quantized_stats=stats, requires_grad=False,
                        device=str(memory_management.get_torch_device()))
                    t = functional_dequantize_4bit(qw)
                    del qw
            with torch.no_grad():
                p.copy_(t.to(p.dtype))
    debug_log("[Ideogram4] TE weights filled in-place (no bulk copies)")
    _QWEN3VL_MODEL = _QWEN3VL_MODEL.to(torch.bfloat16)
    _QWEN3VL_MODEL.eval()
    _QWEN3VL_SOURCE = weight_file
    from backend.patcher.base import ModelPatcher
    _QWEN3VL_PATCHER = ModelPatcher(
        model=_QWEN3VL_MODEL,
        load_device=memory_management.get_torch_device(),
        offload_device=torch.device("cpu"),
    )
    debug_log("[Ideogram4] Encoder weights loaded (residing on CPU; GPU placement decided per-use)")

def _encoder_run_device():
    """Device for encoding.

    Placement follows Forge's OWN built-in text-encoder policy
    (memory_management.text_encoder_device(): --cpu-text-enc flag,
    VRAM-state fallback). No engine-specific env vars.

    Earlier data (2026-10-09, same weights + prompt):
      CPU encode: seed 1619448790 -> card ; earlier seed 981579305 -> clean
      GPU encode: seed 709590072  -> clean
    Result is seed-dependent, NOT device-dependent (Comfy always encodes on
    GPU). When the policy says GPU we load through the VRAM manager and
    fall back to CPU only if there truly is no room (full bf16 TE = 15.1 GB).
    """
    global _QWEN3VL_MODEL
    if memory_management.text_encoder_device().type == "cpu":
        _QWEN3VL_MODEL.to(torch.device("cpu"))
        debug_log("[Ideogram4] encoder on CPU (Forge text-encoder policy)")
        return torch.device("cpu")
    gpu = memory_management.get_torch_device()
    free = memory_management.get_free_memory(gpu)
    need = _QWEN3VL_PATCHER.model_size() + 2 * 1024 * 1024 * 1024  # weights + activation headroom
    try:
        if free >= need:
            memory_management.load_models_gpu([_QWEN3VL_PATCHER], memory_required=2 * 1024 ** 3, force_full_load=True)
            debug_log(f"[Ideogram4] encoder on GPU (free {free/1024**3:.1f} GB, need {need/1024**3:.1f} GB)")
            return _QWEN3VL_PATCHER.current_device
        else:
            # make room once: Forge unloads stale models, then retry full load
            memory_management.unload_all_models()
            free = memory_management.get_free_memory(gpu)
            if free >= need:
                memory_management.load_models_gpu([_QWEN3VL_PATCHER], memory_required=2 * 1024 ** 3, force_full_load=True)
                debug_log(f"[Ideogram4] encoder on GPU after freeing VRAM (free {free/1024**3:.1f} GB)")
                return _QWEN3VL_PATCHER.current_device
            raise torch.OutOfMemoryError  # not enough room -> CPU
    except torch.OutOfMemoryError:
        _QWEN3VL_MODEL.to(torch.device("cpu"))
        memory_management.soft_empty_cache()
        debug_log("[Ideogram4] insufficient VRAM for encoder (full bf16 = 15.1 GB); encoding on CPU")
        return torch.device("cpu")

def encode_qwen3(text_encoder, tokenizer, prompt, device, max_tokens=2048):
    """Run Qwen3-VL text tower using transformers Qwen3VLTextModel.

    Replaces Forge's broken Qwen3_8B forward. Official pipeline uses transformers
    and produces correct features. Abliterated encoder loaded lazily (cached) on CPU.
    Returns: llm_features: (1, L, 53248) float32 tensor
    """
    global _QWEN3VL_MODEL, _QWEN3VL_TOKENIZER
    _load_selected_encoder()

    messages = [{"role": "user", "content": prompt}]
    chat_text = _QWEN3VL_TOKENIZER.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encodings = _QWEN3VL_TOKENIZER(
        chat_text,
        truncation=True,
        max_length=max_tokens,
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    )

    # VRAM-aware placement via Forge memory manager (GPU when room, else CPU)
    try:
        encode_device = _encoder_run_device()
    except Exception as e:
        _QWEN3VL_MODEL.to(torch.device("cpu"))
        memory_management.soft_empty_cache()
        debug_log(f"[Ideogram4] encoder GPU placement failed ({e}); encoding on CPU")
        encode_device = torch.device("cpu")
    target_device = torch.device(device)
    token_ids = encodings["input_ids"].to(encode_device)
    L = token_ids.shape[1]

    # DIAG: log exact tokenization
    try:
        debug_log(
            f"[DIAG tokenize] prompt_type={type(prompt).__name__} prompt={prompt!r} "
            f"n_tokens={L} max_tokens={max_tokens} template={chat_text!r}"
        )
        torch.save(
            {"input_ids": token_ids.cpu(), "template": chat_text, "prompt": prompt},
            "forge_encode_tokens.pt",
        )
    except Exception:
        pass

    TAP_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

    embeds = _QWEN3VL_MODEL.embed_tokens(token_ids)
    from transformers.masking_utils import create_causal_mask
    text_pos = torch.arange(L, device=encode_device).unsqueeze(0)  # (1, L)
    pos_4d = text_pos[None, ...].expand(4, 1, -1)
    mrope_pos = pos_4d[1:]  # (3, 1, L)
    causal_mask = create_causal_mask(
        config=_QWEN3VL_MODEL.config,
        input_embeds=embeds,
        attention_mask=None,
        cache_position=torch.arange(L, device=encode_device),
        past_key_values=None,
        position_ids=text_pos,
    )
    cos_sin = _QWEN3VL_MODEL.rotary_emb(embeds, mrope_pos)

    captured = {}
    h = embeds
    for i, layer in enumerate(_QWEN3VL_MODEL.layers):
        h = layer(
            h,
            attention_mask=causal_mask,
            position_ids=text_pos,
            past_key_values=None,
            position_embeddings=cos_sin,
        )
        if i in TAP_LAYERS:
            captured[i] = h

    # Stack: (num_taps, B, L, H) -> permute -> (B, L, H, num_taps) -> reshape -> (B, L, H*num_taps)
    off = torch.stack([captured[i] for i in TAP_LAYERS], dim=0).permute(1, 2, 3, 0).reshape(1, L, -1)
    del captured, h, cos_sin, causal_mask, embeds

    return off.float().to(target_device)

# --- Engine ---
class Ideogram4(ForgeDiffusionEngine):
    matched_guesses = [model_list.Ideogram4]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False

        clip = CLIP(
            model_dict={"qwen3": huggingface_components["text_encoder"]},
            tokenizer_dict={"qwen3": huggingface_components["tokenizer"]},
        )

        vae = VAE(model=huggingface_components["vae"], is_flux2=True)

        k_predictor = PredictionIdeogram4(estimated_config)

        unet = UnetPatcher.from_model(
            model=huggingface_components["transformer"],
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        # Official guidance_schedule (V4 presets / Comfy CFGOverride [3,0.9,1]):
        # every step at gw EXCEPT the final ~10% which polish at gw=3.
        # Running the polish steps at full UI CFG drifts the model toward its
        # learned safety card. We override CFG on those steps in sigma-space:
        # _polish_sigma is set from our logit-normal sigmas buffer, so the
        # rule holds for any user step count (last 10% of steps = lowest 10%
        # of sigmas). The user's CFG scale acts as the cap.
        def _guidance_polish(args):
            try:
                sigma = args["sigma"]
                s = float(sigma.max().item()) if torch.is_tensor(sigma) else float(sigma)
                if s <= self._k_predictor._polish_sigma:
                    cond_scale = float(args["cond_scale"])
                    eff = min(3.0, cond_scale)
                    return args["uncond_denoised"] + (args["cond_denoised"] - args["uncond_denoised"]) * eff
            except Exception:
                pass
            return args["denoised"]

        unet.set_model_sampler_post_cfg_function(_guidance_polish, disable_cfg1_optimization=True)

        self._text_encoder = huggingface_components["text_encoder"]
        self._tokenizer = huggingface_components["tokenizer"]
        self._k_predictor = k_predictor

        # The TE file the user selected in Forge UI (loader routed it to qwen3_vl).
        global _SELECTED_TE_FILE
        try:
            te_files = getattr(estimated_config, "te_source_files", {}) or {}
            sel = te_files.get("qwen3_vl") or te_files.get("qwen3_8b")
            if sel:
                _SELECTED_TE_FILE = sel
                debug_log(f"[Ideogram4] selected TE file: {sel}")
        except Exception:
            pass

        # Load unconditional LoRA (applied only on negative CFG pass).
        # Preferred location: Forge's standard LoRA dir (models/Lora/Ideogram4/...);
        # if absent, try to download it from HF (ostris/ideogram_4_unconditional_lora).
        try:
            import os
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            lora_path = None
            for cand in (
                os.path.join(root, "models", "Lora", "ideogram4_unconditional.safetensors"),
                os.path.join(root, "models", "Lora", "Ideogram4", "ideogram4_unconditional.safetensors"),
                os.path.join(root, "models", "Lora", "ideogram_4_unconditional_lora", "ideogram_4_unconditional_lora_r16.safetensors"),
                os.path.join(root, "models", "Lora", "Ideogram4", "ideogram_4_unconditional_lora_r16.safetensors"),
                os.path.join(root, "lora_uncond.safetensors"),  # legacy location, last
            ):
                if os.path.isfile(cand):
                    lora_path = cand
                    break
            if lora_path is None:
                debug_log(
                    "[Ideogram4] unconditional LoRA not found in models/Lora; "
                    "downloading from HF (ostris/ideogram_4_unconditional_lora)")
                from huggingface_hub import hf_hub_download
                lora_path = hf_hub_download(
                    repo_id="ostris/ideogram_4_unconditional_lora",
                    filename="ideogram_4_unconditional_lora_r16.safetensors",
                )
                # keep a copy in the standard LoRA dir for future runs
                try:
                    dst_dir = os.path.join(root, "models", "Lora", "Ideogram4")
                    os.makedirs(dst_dir, exist_ok=True)
                    import shutil
                    shutil.copy(lora_path, os.path.join(dst_dir, "ideogram4_unconditional.safetensors"))
                except Exception:
                    pass
            debug_log(f"[Ideogram4] unconditional LoRA source: {lora_path}")
            huggingface_components["transformer"].load_unconditional_lora(lora_path, strength=1.0)
        except Exception as e:
            debug_log(f"[Ideogram4] failed to load unconditional LoRA: {e}")

    def set_clip_skip(self, clip_skip):
        pass

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt):
        # NOTE: we deliberately do NOT load the fp8 JointTextEncoder onto GPU.
        # This engine encodes via the abliterated Qwen3VLTextModel in
        # encode_qwen3(); the component TE is only plumbing for detection/
        # loading. Loading it was a silent 9 GB VRAM waste on every generation.

        # Update logit-normal schedule for actual generation resolution
        width = getattr(prompt, "width", None)
        height = getattr(prompt, "height", None)
        if width and height:
            self._k_predictor.set_resolution(height, width)

        is_negative = getattr(prompt, "is_negative_prompt", False)

        if isinstance(prompt, str):
            prompt = [prompt]

        # Conditioning must live on the sampling device (the fp8 TE is no longer
        # loaded on GPU, so probing its parameters would now return CPU).
        device = memory_management.get_torch_device()
        conds = []
        for text in prompt:
            if is_negative:
                # Official uses ZERO features for unconditional branch
                llm_features = torch.zeros(1, 1, 53248, dtype=torch.float32, device=device)
            else:
                text = normalize_prompt_to_json(text)
                llm_features = encode_qwen3(self._text_encoder, self._tokenizer, text, device)
                try:
                    lf = llm_features.to(torch.float32)
                    debug_log(
                        f"[DIAG encode_qwen3 out] shape={tuple(llm_features.shape)} "
                        f"std={lf.std().item():.4f} absmax={lf.abs().max().item():.4f} "
                        f"frac_zero={(lf == 0).float().mean().item():.4f} "
                        f"device={llm_features.device} dtype={llm_features.dtype}"
                    )
                except Exception:
                    pass
            conds.append(llm_features.squeeze(0))

        # The encoder's job is done. Return its VRAM through the manager's own
        # unload path (moves weights to CPU and updates accounting). A bare
        # unload_model() only drops the bookkeeping and leaves weights on GPU,
        # which starves KModel and forces a broken partial load.
        if not is_negative and _QWEN3VL_PATCHER is not None:
            try:
                memory_management.unload_all_models()
                memory_management.soft_empty_cache()
                _QWEN3VL_MODEL.to(torch.device("cpu"))
                debug_log("[Ideogram4] encoder VRAM released before sampling")
            except Exception as e:
                debug_log(f"[Ideogram4] encoder VRAM release failed: {e}")

        return conds

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self._tokenizer(prompt, truncation=True, max_length=2048)["input_ids"])
        return token_count, max(2048, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)

    def set_resolution(self, height, width, num_steps=None, mu=None, std=None):
        self._k_predictor.set_resolution(height, width, num_steps=num_steps, mu=mu, std=std)

    def set_logit_norm(self, mu, std):
        """Switch the time-shift of the logit-normal schedule without changing
        resolution/steps (used by the Studio presets: official mu=0.0,
        Comfy studio mu=0.5; both std values are model defaults)."""
        kp = self._k_predictor
        kp.set_resolution(kp._height, kp._width, mu=mu, std=std)
        debug_log(f"[Ideogram4] logit-normal schedule set: mu={mu} std={std}")

    # ------------------------------------------------------------------
    # Full unconditional transformer (official two-transformer CFG).
    # Managed through the standard loader pipeline (same dtype/quant logic
    # as the main checkpoint) and the standard memory manager
    # (add_extra_torch_module_during_sampling). The negative CFG branch in
    # Ideogram4Transformer.forward uses it when attached; otherwise it
    # falls back to the unconditional LoRA. Used by the
    # ideogram4_studio extension.
    # ------------------------------------------------------------------

    def _detach_unconditional(self):
        unet = self.forge_objects.unet
        base = unet.model.diffusion_model
        cur = getattr(base, "_uncond_model", None)
        if cur is not None:
            base.set_unconditional_model(None)
            unet.extra_model_patchers_during_sampling = [
                p for p in unet.extra_model_patchers_during_sampling
                if getattr(p.model, "diffusion_model", None) is not cur
            ]
            del cur
            memory_management.soft_empty_cache()

    def set_unconditional_checkpoint(self, path):
        """Attach a full unconditional transformer for the CFG negative
        branch (official pipeline design). None -> revert to LoRA mode.
        Returns the attached path or None."""
        from backend import utils as butils
        from backend.state_dict import load_state_dict, detect_quantization
        from backend.operations import using_forge_operations
        from backend.patcher.unet import UnetPatcher
        from backend.nn.ideogram4 import Ideogram4Transformer, Ideogram4Config

        self._detach_unconditional()
        if path is None:
            debug_log("[Ideogram4] CFG negative branch: unconditional LoRA")
            return None

        sd = butils.load_torch_file(path, safe_load=True)
        if "input_proj.weight" not in sd:
            sd = {k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.")}
        if "input_proj.weight" not in sd:
            raise ValueError(f"{path}: no Ideogram4 transformer weights found")

        unet = self.forge_objects.unet
        base = unet.model.diffusion_model
        unet_config = base.config  # detection-derived dims (dict), same as main load

        load_device = memory_management.get_torch_device()
        offload_device = memory_management.unet_offload_device()
        state_dict_dtype = butils.weight_dtype(sd)
        quant_config = detect_quantization(sd, is_unet=True)
        supported = self.model_config.supported_inference_dtypes

        if quant_config is not None:
            storage_dtype = torch.bfloat16
            debug_log("[Ideogram4] unconditional model: MixedPrecision")
        elif state_dict_dtype in (torch.float8_e4m3fn, torch.float8_e5m2, "nf4", "fp4", "gguf"):
            storage_dtype = state_dict_dtype
            debug_log(f"[Ideogram4] unconditional model: {storage_dtype} (pre-quant)")
        else:
            # same dtype choice as the main model (UI dtype option respected there;
            # keep them identical so both fit the same VRAM plan)
            storage_dtype = base.storage_dtype
            debug_log(f"[Ideogram4] unconditional model: {storage_dtype}")
        computation_dtype = memory_management.inference_cast(
            weight_dtype=storage_dtype, inference_device=load_device, supported_dtypes=supported)

        from transformers.modeling_utils import no_init_weights
        from backend.args import dynamic_args
        dynamic_args.ops = None  # same as loader.py: using_forge_operations re-sets it
        initial_device = memory_management.cpu
        need_manual_cast = storage_dtype != computation_dtype
        with no_init_weights():
            with using_forge_operations(device=initial_device, dtype=storage_dtype,
                                       manual_cast_enabled=need_manual_cast,
                                       sd_dtype=state_dict_dtype, extra_dtype=quant_config):
                model = Ideogram4Transformer(Ideogram4Config.from_dict(unet_config))
        load_state_dict(model, sd)
        model.config = unet_config
        model.storage_dtype = storage_dtype
        model.computation_dtype = computation_dtype
        model.load_device = load_device
        model.initial_device = initial_device
        model.offload_device = offload_device

        # Wrap in a standard KModel/UnetPatcher (same as the main path): the
        # memory manager then sees correct model_size/memory_required and
        # keeps it resident during sampling like any other model (partial
        # load included). Only the raw transformer is used by the negative
        # CFG branch.
        patcher = UnetPatcher.from_model(model=model, diffusers_scheduler=None,
                                         config=self.model_config,
                                         k_predictor=PredictionIdeogram4(self.model_config))
        unet.add_extra_model_patcher_during_sampling(patcher)
        base.set_unconditional_model(model)
        n_params = sum(p.numel() for p in model.parameters())
        debug_log(
            f"[Ideogram4] full unconditional model attached: {path} ({n_params/1e9:.2f}B params)")
        return path