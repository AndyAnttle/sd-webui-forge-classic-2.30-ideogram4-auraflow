import os

import torch

from huggingface_guess import model_list

from backend import memory_management
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.modules.k_prediction import PredictionDiscreteFlow
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.t5_engine import T5TextProcessingEngine

from .spiece_tokenizer import SPieceTokenizer


class _SPieceTokenizerAdapter:
    """Adapts the single-string SPieceTokenizer to the transformers-style
    list-of-texts interface expected by T5TextProcessingEngine.tokenize()."""

    def __init__(self, spiece_tokenizer):
        self.spiece_tokenizer = spiece_tokenizer

    def __call__(self, texts, truncation=False, add_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        input_ids = []
        for text in texts:
            out = self.spiece_tokenizer(text, truncation=truncation, add_special_tokens=add_special_tokens)
            input_ids.append(out["input_ids"])
        return {"input_ids": input_ids}


class AuraFlow(ForgeDiffusionEngine):
    matched_guesses = [model_list.AuraFlow]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False

        # AuraFlow conditions on its own small UMT5 ("pile_t5xl", d_model 2048,
        # vocab 32128) - NOT on t5xxl. The CLIP slot name is internal only;
        # the TE component itself comes pre-filtered from the loader via
        # model_list.AuraFlow.clip_target.
        clip = CLIP(
            model_dict={
                "pile_t5xl": huggingface_components["text_encoder"],
            },
            tokenizer_dict={
                "pile_t5xl": huggingface_components["tokenizer"],
            },
        )

        vae = VAE(model=huggingface_components["vae"])

        k_predictor = PredictionDiscreteFlow(estimated_config)

        unet = UnetPatcher.from_model(
            model=huggingface_components["transformer"],
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config,
        )

        try:
            spiece = SPieceTokenizer.from_pretrained(
                os.path.join(huggingface_components["tokenizer"].name_or_path, "tokenizer.model")
            )
            tokenizer = _SPieceTokenizerAdapter(spiece)
        except ValueError:
            # no spiece file - the HF tokenizer from the bundled config works too
            tokenizer = huggingface_components["tokenizer"]

        self.text_processing_engine_t5 = T5TextProcessingEngine(
            text_encoder=clip.cond_stage_model.pile_t5xl,
            tokenizer=tokenizer,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.use_shift = True

    def set_clip_skip(self, clip_skip):
        pass

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        return self.text_processing_engine_t5(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_t5.tokenize([prompt])[0])
        return token_count, max(255, token_count)

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
