import math

import torch

from backend import args, attention, memory_management
from backend.modules.k_prediction import k_prediction_from_diffusers_scheduler


class KModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, diffusers_scheduler, config, k_predictor):
        super().__init__()

        self.config = config

        self.storage_dtype = model.storage_dtype
        self.computation_dtype = model.computation_dtype

        _store = f"storage: {self.storage_dtype}"
        _compute = f"computation: {self.computation_dtype}"

        if args.dynamic_args.ops.endswith("FP8"):
            _compute += f" + {torch.float8_e4m3fn}"
        if args.dynamic_args.ops.startswith("Mixed"):
            _compute = "computation: Mixed"

        memory_management.logger.info(f"Diffusion Model: {{{_store}, {_compute}}}")

        self.diffusion_model = model
        self.diffusion_model.eval()
        self.diffusion_model.requires_grad_(False)

        if diffusers_scheduler is not None:
            self.predictor = k_prediction_from_diffusers_scheduler(diffusers_scheduler)
        else:
            self.predictor = k_predictor

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        sigma = t
        xc = self.predictor.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        context = c_crossattn
        dtype = self.computation_dtype

        xc = xc.to(dtype)
        t = self.predictor.timestep(t).float()
        context = context.to(dtype)
        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            if hasattr(extra, "dtype"):
                if extra.dtype != torch.int and extra.dtype != torch.long:
                    extra = extra.to(dtype)
            extra_conds[o] = extra

        model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds).float()
        return self.predictor.calculate_denoised(sigma, model_output, x)

    def memory_required(self, input_shape: list[int]) -> float:
        """https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/model_base.py#L354"""
        input_shapes = [input_shape]
        area = sum(map(lambda input_shape: input_shape[0] * math.prod(input_shape[2:]), input_shapes))

        if attention.attention_function is attention.attention_basic:
            # Non-flash attention materializes full score matrices ; coefficient
            # matches the reference implementation's non-flash branch (0.15)
            return (area * 0.16 * self.config.memory_usage_factor) * (1024 * 1024)

        # Flash-like attention (sage / flash / xformers / sdpa / comfy-kitchen int8) never
        # materializes the score matrix. The reference implementation measures that at
        # 0.01 * dtype_size (ComfyUI model_base.memory_required, unchanged since v0.0.5 ;
        # Forge classic 2.29.2 used the same 0.01 here). Neo's initial commit unified the
        # branches as _size * 0.02 : that preserved the non-flash intent (8 * 0.02 = 0.16)
        # but doubled the flash branch as a side effect. For quantized models (activations
        # stay at compute dtype ; only weights are int8/int4) the doubled estimate left no
        # VRAM budget for weights to remain resident during Hires. (upstream #1537)
        # EXPERIMENT (in-progress only): variant C from Speeskees (comment under our
        # commit 36bd7462 / upstream issue #1537): W8A8 models (all quantized linears are
        # int8_tensorwise QuantizedTensors) estimate activations at 1 byte instead of
        # compute-dtype size. Opt-in via --int8-activation-budget 1byte (default: dtype).
        _size = memory_management.dtype_size(self.computation_dtype)
        _coef = 0.01
        if args.args.int8_activation_budget is args.ActivationBudget.ONE_BYTE:
            from backend.quant_ops import QuantizedTensor
            quantized_linears = [m for m in self.diffusion_model.modules() if getattr(m, "quant_format", None) is not None and hasattr(m, "in_features") and hasattr(m, "out_features")]
            if quantized_linears and all(
                m.quant_format == "int8_tensorwise"
                and isinstance(getattr(m, "weight", None), QuantizedTensor)
                and not getattr(m, "weight_function", ())
                for m in quantized_linears
            ):
                if not getattr(self, "_memguard_logged", False):  # one short line per model instance
                    self._memguard_logged = True
                    memory_management.logger.info(f"[memguard-exp] W8A8 model: {len(quantized_linears)} int8 linears; estimating activations at 1 byte instead of {_size}")
                _size = 1
        return (area * _size * _coef * self.config.memory_usage_factor) * (1024 * 1024)
