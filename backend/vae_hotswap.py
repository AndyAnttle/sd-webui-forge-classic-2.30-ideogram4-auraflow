"""X/Y/Z 'VAE' axis support (upstream №1547 research): latent-interface classifier and VAE hot-swap.

Models and VAE files only exchange latents through a small interface: latent channel
count, spatial ratio, and how the model normalizes them (latent_format, injected from
the model config, not from the file). The strict in-place weight load, however, only
works when a file shares the exact tensor graph of the built first_stage_model.

This module classifies an override VAE file against the live model:

  "inplace"      file graph (after the loader's own format conversions: diffusers
                 layout, flux1 conv aliasing) equals the built VAE graph - strict
                 weight swap, the existing fast path.
  "hotswap"      the latent interfaces match (channels/ratio/normalization family)
                 but the internals differ (2D vs 3D layout, renamed key families):
                 build only a new VAE module through the model's own loader branch
                 and patch it into the live engine. The UNet/CLIP and the model
                 memory entries are untouched; restoring re-attaches the original
                 wrapper. Used for X/Y/Z VAE axes across 2D/3D VAE families.
  "incompatible" the file cannot ever serve this model (different latent channels
                 or ratio, e.g. a z64 Qwen-Image 2.1 VAE for a z16 model): raise a
                 clear per-cell message instead of a silent black cell.
  "rebuild"      everything the classifier does not positively understand
                 (GGUF files, exotic families, failed dry-build): the previous
                 model-rebuild path, unchanged.
"""
import os

from backend import memory_management

INPLACE = "inplace"
HOTSWAP = "hotswap"
INCOMPATIBLE = "incompatible"
REBUILD = "rebuild"

# hot-swap state: the original wrapper is kept (CPU-resident) so restore costs no reload
_hotswap_state: dict = None


def _graph(sd: dict) -> dict:
    return {k: tuple(v.shape) for k, v in sd.items() if not k.startswith("loss")}


def normalized_file_sd(vae_sd: dict) -> dict:
    """Apply the same conversions backend/loader.replace_state_dict applies before VAE
    class selection, so convertible-format files (diffusers layout, flux1 conv aliasing)
    can also reach the in-place fast path."""
    from modules_forge.packages.huggingface_guess.utils import state_dict_prefix_replace

    sd = state_dict_prefix_replace(vae_sd, {"decoder.post_quant_conv.": "post_quant_conv.", "encoder.quant_conv.": "quant_conv."})
    if "decoder.up_blocks.0.resnets.0.norm1.weight" in sd:
        from modules_forge.packages.huggingface_guess.diffusers_convert import convert_vae_state_dict

        sd = convert_vae_state_dict(sd)
    return sd


def file_interface(vae_sd: dict):
    """Derive (latent_channels, spatial ratio, family) from a VAE file's state dict.
    All rules verified against real headers of the whole VAE library (study_1547_*.py).
    Decoder-side channels are the reliable source: encoder conv_out stores 2x channels
    on some families and flux-family stores the packed raw width."""
    g = _graph(vae_sd)

    def ch(k, i):
        return g[k][i] if k in g else None

    if "encoder.conv1.weight" in g and len(g["encoder.conv1.weight"]) == 5:  # 3D wan-style
        z = ch("decoder.conv1.weight", 1)
        img_ch = ch("decoder.head.2.weight", 0)
        if z == 64 or img_ch == 4:
            return {"z": 64, "ratio": 16, "family": "qwen21", "layout": "3d"}
        if z == 16:
            return {"z": 16, "ratio": 8, "family": "wan21", "layout": "3d"}
        return None
    z = ch("decoder.conv_in.weight", 1)
    if z is not None:
        if z == 16:
            if "post_quant_conv.weight" in g:
                return {"z": 16, "ratio": 8, "family": "wan21", "layout": "2d"}  # Qwen2D
            return {"z": 16, "ratio": 8, "family": "flux1", "layout": "2d"}  # FLUX.1 / Chroma / Z-Image ae
        if z == 32:
            return {"z": 128, "ratio": 16, "family": "flux2", "layout": "2d-packed"}  # FLUX.2 / Ideogram4 raw storage
        if z == 4:
            return {"z": 4, "ratio": 8, "family": "sd", "layout": "2d"}  # SD15/SDXL-class
    return None


def model_interface(model):
    """The live engine's latent interface from its own VAE wrapper + model config."""
    vae = model.forge_objects.vae
    norm = type(model.model_config.latent_format).__name__
    family = {
        "Wan21": "wan21",
        "QwenImage21": "qwen21",
        "Flux": "flux1",
        "Flux2": "flux2",
        "Ideogram4": "flux2",  # same raw storage (128 packed); normalization is model-side either way
    }.get(norm, None)
    ratio = vae.upscale_ratio if isinstance(vae.upscale_ratio, int) else 8
    return {"z": vae.latent_channels, "ratio": ratio, "family": family}


def incompatible_message(model, vae_file: str, f_ifc, m_ifc) -> str:
    return (
        f"{os.path.basename(str(vae_file))} cannot serve the loaded model on the VAE axis: it is a "
        f"{f_ifc['z']}-channel (ratio {f_ifc['ratio']}, {f_ifc['layout']}) VAE while the model uses "
        f"{m_ifc['z']}-channel latents (ratio {m_ifc['ratio']}). VAE files may only be switched between "
        f"files of the model's own latent format (same interchangeability as the VAE / Text Encoder dropdown)."
    )


def _build_evidence(model, vae_sd: dict):
    """Ask the model's OWN construction branch to build+load the file (CPU, throwaway).
    Returns True (the branch accepts it - the interface table was wrong), a reason string
    (the branch rejects it - the incompatibility is proven, not a table opinion),
    or None (construction infra unavailable - e.g. unit tests with fake models)."""
    guess = getattr(model, "model_config", None)
    repo = getattr(guess, "huggingface_repo", None)
    if guess is None or not repo:
        return None
    try:
        from diffusers import DiffusionPipeline

        from backend.loader import HF, load_huggingface_component

        cfg = DiffusionPipeline.load_config(os.path.join(HF, repo))
        entry = cfg.get("vae")
        if not isinstance(entry, list) or len(entry) != 2:
            return None
        module = load_huggingface_component(guess, "vae", entry[0], entry[1], os.path.join(HF, repo), vae_sd)
        return module is not None
    except Exception as e:
        return " ".join(str(e).split())[:300]


def classify(model, vae_sd: dict, vae_file: str):
    """Decision for an override VAE file against the live model. Returns
    (kind, message_or_None). Never raises for unclassifiable inputs: those stay on
    the proven rebuild path."""
    built = model.first_stage_model.state_dict()
    model_graph = _graph(built)

    def filtered(sd):
        return {k: v for k, v in sd.items() if k[0:4] != "loss" and k not in {"model_ema.decay", "model_ema.num_updates"}}

    normalized = filtered(vae_sd)
    try:
        normalized = normalized_file_sd(filtered(vae_sd))
    except Exception:
        pass

    if _graph(normalized) == model_graph or _graph(filtered(vae_sd)) == model_graph:
        return INPLACE, None

    f_ifc = file_interface(normalized)
    m_ifc = model_interface(model)
    if f_ifc is None or m_ifc["family"] is None:
        return REBUILD, None  # GGUF / unusual family / unusual model: previous path

    if f_ifc["z"] != m_ifc["z"] or f_ifc["ratio"] != m_ifc["ratio"] or f_ifc["family"] != m_ifc["family"]:
        # do not trust the table alone: the model's own construction branch has the final say.
        evidence = _build_evidence(model, normalized)
        if evidence is True:
            return HOTSWAP, None  # branch accepts the file - hot-swap it and prove at runtime
        reason = None if evidence is None else evidence  # None: fake model (tests) - table verdict only
        msg = incompatible_message(model, vae_file, f_ifc, m_ifc)
        if reason:
            msg += f" (construction branch: {reason})"
        return INCOMPATIBLE, msg

    return HOTSWAP, None


def hotswap_vae(model, vae_file: str, vae_sd: dict) -> bool:
    """Replace only model.forge_objects.vae with a freshly built module for this file,
    using the model's own model_index VAE branch (exact build semantics of the
    dropdown). Returns True if a swap was performed (False when already swapped to
    the same file). The original wrapper is retained for restore_hotswap()."""
    global _hotswap_state

    if _hotswap_state is not None and _hotswap_state["file"] == str(vae_file):
        return False

    from backend.loader import HF, load_huggingface_component
    from backend.patcher.vae import VAE
    from diffusers import DiffusionPipeline

    guess = model.model_config
    local_path = os.path.join(HF, guess.huggingface_repo)
    pipeline_config = DiffusionPipeline.load_config(local_path)
    vae_entry = pipeline_config.get("vae")
    if not isinstance(vae_entry, list) or len(vae_entry) != 2:
        raise RuntimeError("model has no VAE entry in its model_index; hot-swap unavailable")

    module = load_huggingface_component(guess, "vae", vae_entry[0], vae_entry[1], local_path, vae_sd)
    if module is None:
        raise RuntimeError("model loader branch declined the VAE file; hot-swap unavailable")
    module.latent_format = guess.latent_format

    old_vae = model.forge_objects.vae
    new_vae = VAE(model=module, device=old_vae.device, dtype=old_vae.vae_dtype, is_wan=old_vae.is_wan, is_qwen21=(old_vae.latent_channels == 64))

    memory_management.unload_model(old_vae.patcher)  # drop from GPU registry; weights kept in the retained wrapper

    _hotswap_state = {"file": str(vae_file), "vae": old_vae}
    for fo in (model.forge_objects, model.forge_objects_original, model.forge_objects_after_applying_lora):
        if fo is not None and fo.vae is old_vae:
            fo.vae = new_vae

    # same latent interface by construction: opt_f, vae approx/taesd and LoRA snapshots stay valid
    return True


def restore_hotswap(model) -> bool:
    """Re-attach the retained original VAE wrapper; costs no model reload."""
    global _hotswap_state
    if _hotswap_state is None:
        return False
    original = _hotswap_state["vae"]
    current = model.forge_objects.vae
    if current is not original:
        memory_management.unload_model(current.patcher)
        del current
    for fo in (model.forge_objects, model.forge_objects_original, model.forge_objects_after_applying_lora):
        if fo is not None:
            fo.vae = original
    _hotswap_state = None
    return True


def hotswap_active() -> bool:
    return _hotswap_state is not None
