# Prompt Group Pre-Encode (upstream feature request #1435)
#
# When n_iter > 1 with heavy text encoders, the WebUI currently re-loads and
# re-encodes the text encoder once per batch:  load TE -> encode batch -> unload
# TE -> sample -> repeat. This script encodes a configurable group of upcoming
# batches in one go while the TE is already resident, then feeds those results
# back through the core's own cond-cache mechanism (get_conds_with_caching), so
# batches covered by a cached group never touch the TE again.
#
# Mechanism (no core monkey-patching):
#   - batch 0, before setup_conds(): for each upcoming batch k in the group,
#     temporarily point the processing object at batch k's prompt slice, run the
#     core's own p.setup_conds() (identical encoding semantics, same caches),
#     snapshot the resulting cache triples (params, conds, extra_generation_params),
#     then restore batch 0's state untouched.
#   - batches 1..G-1: inject the snapshot into p.cached_c / p.cached_uc just
#     before the core's setup_conds() runs; get_conds_with_caching hits it because
#     the parameter tuple matches, and encoding is skipped entirely.
#   - Any state change a script made to prompts/params in the meantime makes the
#     tuple mismatch -> plain cache miss -> normal (uncached) encoding. The
#     feature can therefore only save work, never change results.
#   - Hires./Refiner passes use their own parameter tuples (different sizes/model)
#     and keep their normal per-pass encoding; only base-pass conds are grouped.

import gradio as gr

from modules import scripts, shared


class PromptGroupPreEncode(scripts.Script):
    sorting_priority = 3  # near the end of process_batch order: inject after other scripts ran

    def title(self):
        return "Prompt Group Pre-Encode"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(False, label="Pre-encode upcoming batches while the text encoder is loaded")
            group_size = gr.Slider(
                minimum=2, maximum=32, step=1, value=5,
                label="Batch group size (how many batches to encode in one pass)",
            )
        return [enabled, group_size]

    def process_batch(self, p, enabled=False, group_size=0, *, batch_number: int, **kwargs):
        if not isinstance(getattr(p, "prompts", None), list):
            return

        if batch_number == 0:
            if bool(enabled) and int(group_size) >= 2:
                self._preencode(p, int(group_size))
            return

        store = getattr(p, "_forge_preencode", None)
        if store and batch_number in store:
            cached_c, cached_uc = store[batch_number]
            if cached_c is not None:
                p.cached_c = list(cached_c)
            if cached_uc is not None:
                p.cached_uc = list(cached_uc)

    # ------------------------------------------------------------------
    def _preencode(self, p, size: int):
        if p.n_iter < 2 or p.is_hr_pass:
            return
        try:
            from backend.args import dynamic_args
            if getattr(dynamic_args, "kontext", False):
                return  # stateful reference latents; group encoding would change semantics
        except Exception:
            pass
        groups = min(size, p.n_iter)
        bs = p.batch_size

        # current batch state, restored verbatim after the pre-encode sweep
        saved_prompts = list(p.prompts)
        saved_negative = list(p.negative_prompts)
        saved_extra = getattr(p, "extra_network_data", None)

        store = {}
        for k in range(1, groups):
            lo, hi = k * bs, (k + 1) * bs
            if hi > len(p.all_prompts):
                break
            p.prompts = list(p.all_prompts[lo:hi])
            p.negative_prompts = list(p.all_negative_prompts[lo:hi])
            p.parse_extra_network_prompts()

            # encoding runs under the currently active model state; a batch whose
            # extra-network tags differ from this batch's would encode under the wrong
            # network state — stop the group at that batch instead
            try:
                if dict(getattr(p, "extra_network_data", {}) or {}) != dict(saved_extra or {}):
                    break
            except Exception:
                break

            p.clear_prompt_cache()
            # keep other groups' extra_generation_params from leaking into batch 0's infotext
            shared.sd_model.extra_generation_params = {}

            try:
                p.setup_conds()
            except Exception:
                # a failed pre-encode must never break the generation itself
                store = {}
                break

            store[k] = (list(p.cached_c), list(p.cached_uc))

        p.prompts = saved_prompts
        p.negative_prompts = saved_negative
        p.extra_network_data = saved_extra
        p.clear_prompt_cache()
        shared.sd_model.extra_generation_params = {}

        if store:
            p._forge_preencode = store
            shared.sd_model.comments.append(
                f"Prompt Group Pre-Encode: batches 1..{max(store)} pre-encoded in one group."
            )
