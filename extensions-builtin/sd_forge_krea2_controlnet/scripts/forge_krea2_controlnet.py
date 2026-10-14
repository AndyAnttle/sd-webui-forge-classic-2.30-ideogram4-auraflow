"""Krea2 Control-LoRA support (upstream issue #1507).

Krea-2 control conditioning (e.g. https://huggingface.co/Patil/Krea-2-depth-controlnet)
is NOT a classic ControlNet: it is a rank-64 LoRA over the Krea2 DiT blocks plus an
expanded input projection ("first", in_features = image_features + control_features)
whose control half receives tokens of a VAE-encoded control image.

This patcher mirrors the reference ComfyUI implementation
(https://github.com/facok/comfyui-krea2-controlnet):
  - block LoRA weights are applied as normal Forge LoRA patches (unit weight = strength),
  - the control projection half is injected into the DiT `first` only while a control
    latent is attached; the original `first` is restored immediately after every call,
  - the control image is encoded with the model's own VAE into the model's latent space,
    exactly like the Krea2 engine does for its edit references (krea.py encode_vision).

Limitations (v1): control applies over the full guidance range (start/end percents are
not honored); control image is always normalized per-image minmax like the reference
default `normalize=per_image_minmax`; `invert` is not exposed (the Depth Anything output
convention matches the trained convention, near objects white).
"""

import logging

import torch
import torch.nn.functional as F

from backend.logging import setup_logger
from backend.misc.image_resize import adaptive_resize
from backend.nn.krea import SingleStreamDiT
from backend.sampling.condition import repeat_to_batch_size
from backend.utils import pad_to_patch_size
from modules_forge.packages.comfy.weight_adapter.lora import LoRAAdapter
from modules_forge.shared import add_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher

logger = logging.getLogger("krea2-control")
setup_logger(logger)


class ControlFirstProxy(torch.nn.Module):
    """Wraps the model's real `first` projection and adds the control-token half.

    Mirrors Krea2ControlInputProjection.forward from the reference (the branch that
    keeps `original_first` for image tokens and only adds `control @ Wc`). The model's
    own `first` is never modified: this proxy exists only for the duration of one
    model call, so LoRA patches, offloading and state_dict views are untouched.
    """

    def __init__(self, original: torch.nn.Module, control_weight: torch.Tensor, control_tokens: torch.Tensor):
        super().__init__()
        self.original = original
        self.control_weight = control_weight
        self.control_tokens = control_tokens

    def forward(self, img_tokens):
        out = self.original(img_tokens)
        control_tokens = self.control_tokens.to(device=out.device, dtype=out.dtype)
        control_weight = self.control_weight.to(device=out.device, dtype=out.dtype)
        return out + F.linear(control_tokens, control_weight, None)


class Krea2ControlPatcher(ControlModelPatcher):

    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        first_weight = state_dict.get("first.weight", None)
        if first_weight is None or first_weight.ndim != 2 or first_weight.shape[1] % 2 != 0:
            return None
        has_block_pairs = any(
            k.startswith("blocks.") and k.endswith(".A") and f"{k[:-2]}.B" in state_dict
            for k in state_dict
        )
        if not has_block_pairs:
            return None
        return Krea2ControlPatcher(state_dict)

    def __init__(self, state_dict):
        super().__init__()
        self.state_dict = state_dict

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet
        kmodel = unet.model
        dit = getattr(kmodel, "diffusion_model", None)

        if not isinstance(dit, SingleStreamDiT):
            logger.error("Krea2 Control model needs a Krea2 checkpoint selected — skipped.")
            return

        first_weight = self.state_dict["first.weight"]
        image_features = dit.first.in_features
        if first_weight.shape[1] != image_features * 2 or first_weight.shape[0] != dit.first.out_features:
            logger.error(
                f"Krea2 Control projection does not fit this Krea2 model: "
                f"file first.weight {tuple(first_weight.shape)} vs model {dit.first.out_features}x{image_features}."
            )
            return

        control_weight = first_weight[:, image_features:].float().cpu()

        # ---- block LoRA patches: "<base>.A"/"<base>.B" -> diffusion_model.<base>.weight
        model_keys = set(k for k in unet.model.state_dict() if k.startswith("diffusion_model."))
        patches = {}
        skipped = 0
        for k in self.state_dict:
            if not k.startswith("blocks.") or not k.endswith(".A"):
                continue
            base = k[:-2]
            b_key = f"{base}.B"
            if b_key not in self.state_dict:
                continue
            target = f"diffusion_model.{base}.weight"
            if target not in model_keys:
                skipped += 1
                continue
            a = self.state_dict[k]  # down [rank, in]
            b = self.state_dict[b_key]  # up [out, rank]
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                skipped += 1
                continue
            alpha = a.shape[0]  # the file carries no alpha key; rank-as-alpha convention (matches reference)
            patches[target] = LoRAAdapter({k, b_key}, (b, a, float(alpha), None, None, None))

        if not patches:
            logger.error("Krea2 Control file has no block weight pairs compatible with this model — skipped.")
            return

        new_unet = unet.clone()
        loaded = new_unet.add_patches(patches, strength_patch=self.strength)
        if not loaded:
            logger.error("Krea2 Control LoRA patches were rejected by the model — skipped.")
            return
        logger.info(f"Krea2 Control LoRA: merged {len(loaded)}/{len(patches)} block patches at weight {self.strength}" + (f" (skipped {skipped})" if skipped else ""))

        # ---- control image: grayscale + per-image minmax to [0,1] (VAE's process_input does [-1,1])
        control_image = cond.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        flat = control_image.flatten(1)
        lo = flat.amin(1).view(-1, 1, 1, 1)
        hi = flat.amax(1).view(-1, 1, 1, 1)
        control_image = (control_image - lo) / (hi - lo + 1e-6)

        vae = process.sd_model.forge_objects.vae
        with torch.inference_mode():
            latent = vae.encode(control_image.movedim(1, -1))
            latent = vae.first_stage_model.process_in(latent)
        if latent.ndim == 5:
            latent = latent[:, :, 0]  # Wan-style time dim; single-frame control
        control_latent = latent

        patch = dit.patch

        def control_tokens_from_latent(x):
            x4 = x
            if x4.ndim == 5:
                if x4.shape[2] != 1:
                    raise RuntimeError("Krea2 Control does not support multi-frame latents yet.")
                x4 = x4[:, :, 0]
            ctrl = repeat_to_batch_size(control_latent, x4.shape[0])
            ctrl = adaptive_resize(ctrl, x4.shape[-1], x4.shape[-2], "bilinear", "center")
            ctrl = pad_to_patch_size(ctrl, (patch, patch), padding_mode="replicate").to(device=x.device, dtype=x.dtype)
            b, c, h, w = ctrl.shape
            features = c * patch * patch
            if features != image_features:
                raise RuntimeError(f"Krea2 Control tokens give {features} features, model expects {image_features}.")
            return ctrl.reshape(b, c, h // patch, patch, w // patch, patch).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // patch) * (w // patch), features)

        def krea2_control_wrapper(model_function, executor_args):
            x = executor_args["input"]
            timestep = executor_args["timestep"]
            c = executor_args["c"]

            bound = getattr(model_function, "__self__", None)
            live_dit = getattr(bound, "diffusion_model", None)
            first = getattr(live_dit, "first", None)
            if live_dit is None or first is None or isinstance(first, ControlFirstProxy):
                return model_function(x, timestep, **c)

            tokens = control_tokens_from_latent(x)
            live_dit.first = ControlFirstProxy(first, control_weight, tokens)
            try:
                return model_function(x, timestep, **c)
            finally:
                live_dit.first = first

        new_unet.set_model_unet_function_wrapper(krea2_control_wrapper)
        process.sd_model.forge_objects.unet = new_unet


add_supported_control_model(Krea2ControlPatcher)
