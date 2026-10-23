"""LLaVA-OneVision runtime/checkpoint names used by Transformers 4.53."""

CONVERSION = {
    "^language_model.model": "model.language_model",
    "^vision_tower": "model.vision_tower",
    "^multi_modal_projector": "model.multi_modal_projector",
    "^image_newline": "model.image_newline",
    "^language_model.lm_head": "lm_head",
}


def runtime_to_checkpoint(key):
    for checkpoint, runtime in CONVERSION.items():
        if key == runtime or key.startswith(runtime + "."):
            return checkpoint.removeprefix("^") + key[len(runtime):]
    return key


def verify_installed_mapping():
    from transformers import LlavaOnevisionForConditionalGeneration

    if LlavaOnevisionForConditionalGeneration._checkpoint_conversion_mapping != CONVERSION:
        raise ValueError("Installed LLaVA checkpoint conversion differs from the verified mapping")
