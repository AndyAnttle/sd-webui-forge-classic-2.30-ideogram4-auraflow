"""Ideogram 4 Studio - region/bbox editor + unconditional control for Ideogram4.

Collapsed accordion = off (same pattern as sd_forge_radial / forge extensions).
While open:
* unconditional control: unconditional LoRA (default, engine-loaded) or a full
  unconditional transformer discovered near the current checkpoint by the
  "unconditional" name part (official pipeline design - two transformers);
* the studio widget itself (toolbar, canvas, element panel, inline typing,
  undo/redo, groups, palette) is rendered entirely by
  javascript/ideogram4_studio.js, a port of the ComfyUI ideogram-helper
  studio. Its state is pushed into a hidden Textbox (the standard Forge
  pattern used by modules_forge.forge_canvas), consumed here.

The studio JSON is turned into the official caption JSON (the schema
accepted by Ideogram4 itself, cf. ideogram4-official caption_verifier) and
injected into the prompt before encoding.
"""

import json
import os

import gradio as gr

import modules.shared as shared
import modules.sd_models as sd_models
from modules import scripts
from modules.ui_components import InputAccordion

MODE_LORA = "Unconditional LoRA (default)"
MODE_FULL = "Full unconditional checkpoint (recommended)"

MODE_CHOICES = [MODE_LORA, MODE_FULL]

UNCOND_FIELD = "Unconditional model control:"


def find_unconditional_checkpoints(current_checkpoint: str) -> list:
    """Standard Forge checkpoint registry (modules.sd_models.checkpoints_list),
    filtered by the name part that both official and community unconditional
    releases carry. Checkpoints stored in the same folder as the main model
    rank first."""
    infos = [ci for ci in sd_models.checkpoints_list.values()
             if "unconditional" in ci.title.lower()]
    current_info = None
    if current_checkpoint:
        current_info = sd_models.get_closet_checkpoint_match(current_checkpoint)

    def rank(ci):
        near = 0 if (current_info is not None and
                     os.path.dirname(ci.filename).lower() == os.path.dirname(current_info.filename).lower()) else 1
        return (near, ci.title.lower())

    return sorted((ci.title for ci in infos),
                  key=lambda t: rank(sd_models.checkpoints_list[t]) if t in sd_models.checkpoints_list else (1, t))


def _clean_palette(pal):
    """#RRGGBB uppercase only, max 5 (official element_palette_max)."""
    out = []
    for c in (pal or [])[:5]:
        if isinstance(c, str) and len(c) == 7 and c[0] == "#":
            h = c[1:].upper()
            if all(ch in "0123456789ABCDEF" for ch in h):
                out.append("#" + h)
    return out


def build_caption(state: dict):
    """Studio JSON -> caption JSON that passes the OFFICIAL CaptionVerifier
    (ideogram4-official/src/ideogram4/caption_verifier.py):
    - bbox = [ymin, xmin, ymax, xmax] ints in 0..1000 (studio stores this order)
    - element key order: type, bbox, [text], desc, color_palette
    - background = string; style_description: photo XOR art_style, fixed order
    """
    els = []
    for el in state.get("elements") or []:
        bb = el.get("bbox")
        if not bb:
            continue  # never-drawn element: not a region, skip
        bb = [max(0, min(1000, int(v))) for v in bb[:4]]
        if len(bb) != 4 or bb[0] > bb[2] or bb[1] > bb[3]:
            continue  # degenerate box, would only produce verifier warnings
        if el.get("type") == "text":
            item = {"type": "text", "bbox": bb}
            item["text"] = el.get("text") or ""
            if el.get("desc"):
                item["desc"] = el["desc"]
        else:
            item = {"type": "obj", "bbox": bb}
            item["desc"] = el.get("desc") or ""
        pal = _clean_palette(el.get("color_palette"))
        if pal:
            item["color_palette"] = pal
        els.append(item)
    if not els:
        return None
    background = (state.get("background") or "").strip() \
        or "a coherent scene matching the elements"
    # official pattern (and engine normalize): keep a full-canvas background element
    def _is_full(b):
        return isinstance(b, list) and len(b) == 4 and b[0] <= 0 and b[1] <= 0 and b[2] >= 1000 and b[3] >= 1000
    if not any(_is_full(e.get("bbox")) for e in els):
        bg = {"type": "obj", "bbox": [0, 0, 1000, 1000], "desc": background}
        els.insert(0, bg)
    style = state.get("style") or {}
    medium = (style.get("medium") or "photograph").strip() or "photograph"
    aesthetics = style.get("aesthetics") or ""
    lighting = style.get("lighting") or ""
    if "photo" in medium.lower():
        style_description = {
            "aesthetics": aesthetics,
            "lighting": lighting,
            "photo": style.get("photo") or "high resolution",
            "medium": medium,
        }
    else:
        style_description = {
            "aesthetics": aesthetics,
            "lighting": lighting,
            "medium": medium,
            "art_style": style.get("art_style") or "digital art",
        }
    caption = {
        "high_level_description": (state.get("high_level_description") or "").strip()
        or "A scene composed of the elements described below.",
        "style_description": style_description,
        "compositional_deconstruction": {
            "background": background,
            "elements": els,
        },
    }
    return caption


def parse_state(raw):
    if not raw or not isinstance(raw, str) or not raw.strip():
        return None
    try:
        state = json.loads(raw)
    except Exception:
        return None
    if not isinstance(state, dict):
        return None
    return state


class Script(scripts.ScriptBuiltinUI):
    sorting_priority = 18136

    def title(self):
        return "Ideogram 4 Studio"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "i2i" if is_img2img else "t2i"
        NOTE_MISSING = (
            "No unconditional checkpoint found. Recommended: download the official "
            "Ideogram4 unconditional release and place it in the same folder as the "
            "main Ideogram4 model, then press Refresh.")

        with InputAccordion(False, label=self.title()) as enabled:
            gr.Markdown(
                "Draw regions on the canvas, **double-click** a region to type its prompt, "
                "right-click for the menu, arrow keys nudge, Del removes. The scene panel "
                "below the canvas holds shared fields (description, background, colors, style). "
                "While this accordion is collapsed, the normal plain prompt + unconditional "
                "LoRA is used, exactly as before.")
            initial_found = find_unconditional_checkpoints(getattr(shared.opts, "sd_model_checkpoint", "") or "")
            uncond_mode = gr.Radio(
                choices=MODE_CHOICES if initial_found else [MODE_LORA],
                value=MODE_LORA, label=UNCOND_FIELD,
                info="Full unconditional transformer = official two-model CFG; "
                     "much stronger at suppressing the safety-refusal card.")
            with gr.Row(visible=False) as full_row:
                uncond_choice = gr.Dropdown(
                    choices=initial_found,
                    value=initial_found[0] if initial_found else None,
                    label="Unconditional checkpoint",
                    info="Discovered by the 'unconditional' name part; checkpoints next to "
                         "the current model are listed first.")
                uncond_refresh = gr.Button("Refresh list", scale=0)
            uncond_note = gr.Markdown("" if initial_found else NOTE_MISSING,
                                      visible=not initial_found)
            # hidden state channel written by the JS studio (forge_canvas pattern)
            studio_state = gr.Textbox(
                elem_id=f"id4s_state_{tab}", label="id4s_state",
                visible=False, value="")
            # the JS studio renders its whole UI inside this slot
            gr.HTML(elem_id=f"id4s_canvas_{tab}")

        for comp in (enabled, uncond_mode, uncond_choice, uncond_refresh,
                     uncond_note, studio_state):
            comp.do_not_save_to_config = True

        def switch_mode(mode):
            return gr.update(visible=(mode == MODE_FULL))

        def refresh():
            """(Re)discover unconditional checkpoints; the full-model option
            exists in the UI only while at least one was found."""
            found = find_unconditional_checkpoints(getattr(shared.opts, "sd_model_checkpoint", "") or "")
            return (
                gr.update(choices=MODE_CHOICES if found else [MODE_LORA],
                          value=MODE_LORA),
                gr.update(choices=found, value=found[0] if found else None),
                gr.update(value="" if found else NOTE_MISSING, visible=not found),
                gr.update(visible=False),  # row hides until Full mode is selected again
            )

        uncond_mode.change(fn=switch_mode, inputs=[uncond_mode],
                           outputs=[full_row], queue=False)
        uncond_refresh.click(fn=refresh,
                             inputs=[], outputs=[uncond_mode, uncond_choice,
                                                 uncond_note, full_row],
                             queue=False)

        return enabled, uncond_mode, uncond_choice, studio_state

    def setup(self, p, enabled, uncond_mode, uncond_choice, studio_state):
        """Sampler/scheduler/steps/CFG are NOT overridden here: the studio
        preset button writes them into the normal Forge UI, and generation
        follows the UI (the user stays free to adjust). The only thing the
        preset marker carries is the logit-normal time-shift, which has no
        Forge control of its own.

        Changing sampler/scheduler/steps/CFG by hand does NOT clear the preset:
        the time-shift may benefit other samplers too, so mu stays while the
        preset itself is selected (it is cleared by choosing the other preset
        or toggling it off / closing the studio)."""
        if not enabled:
            self._preset = None
            return
        state = parse_state(studio_state)
        self._preset = (state or {}).get("preset")

    def apply_time_shift(self, enabled, preset):
        """Keep the logit-normal time-shift in sync with the selected preset.
        Called even when the studio is off or the preset was cleared: a shift
        applied earlier must NOT linger on the loaded model - switching to
        another preset / manual sampler settings / another model resets it."""
        try:
            from backend.diffusion_engine.ideogram4 import Ideogram4
        except Exception:
            return
        try:
            if not isinstance(shared.sd_model, Ideogram4):
                # different model: it has its own fresh defaults; forget ours
                self._mu_applied = None
                return
            if enabled and preset == "official":
                mu, std = 0.0, 1.75
            elif enabled and preset == "comfy":
                mu, std = 0.5, 1.75
            else:
                if getattr(self, "_mu_applied", None) is None:
                    return  # we never changed anything -> leave the engine alone
                mu, std = 0.0, 1.75  # the engine's own default
            shared.sd_model.set_logit_norm(mu, std)
            self._mu_applied = (preset if enabled and preset in ("official", "comfy") else None)
        except Exception as e:
            from modules import errors
            errors.report(f"[Ideogram4 Studio] cannot set logit-normal: {e}")

    def process(self, p, enabled, uncond_mode, uncond_choice, studio_state):
        state = parse_state(studio_state)

        # 0) time-shift from the preset - or back to engine defaults when no
        # preset is selected (runs also when the studio is switched off)
        preset = None
        if enabled:
            preset = getattr(self, "_preset", None) or (state or {}).get("preset")
        self.apply_time_shift(enabled, preset)

        if not enabled:
            return

        # 1) unconditional control - only for a currently loaded Ideogram4 engine
        try:
            from backend.diffusion_engine.ideogram4 import Ideogram4
            engine = shared.sd_model if isinstance(shared.sd_model, Ideogram4) else None
        except Exception:
            engine = None
        if engine is not None:
            if uncond_mode == MODE_FULL and uncond_choice:
                ci = sd_models.get_closet_checkpoint_match(uncond_choice)
                if ci is not None:
                    try:
                        engine.set_unconditional_checkpoint(ci.filename)
                    except Exception as e:
                        from modules import errors
                        errors.report(f"[Ideogram4 Studio] cannot attach unconditional model: {e}")
            else:
                engine.set_unconditional_checkpoint(None)

        # 2) studio regions -> official caption injected into every prompt ----------
        state = parse_state(studio_state)
        if state is None:
            return
        caption = build_caption(state)
        if caption is None:
            return
        payload = json.dumps(caption, ensure_ascii=False)
        # the engine passes prompts that already look like JSON straight to the
        # tokenizer; studio output is authoritative, so replace wholesale.
        # all_prompts is (re)sliced into p.prompts per batch, so this covers
        # every batch and iteration.
        if isinstance(p.all_prompts, list):
            p.all_prompts = [payload for _ in p.all_prompts]
        else:
            p.all_prompts = [payload]
