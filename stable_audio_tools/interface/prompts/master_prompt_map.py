import re

from . import edm_elements_prompts, piano_prompts, vocal_textures_prompts
from .foundation_prompts import prompt_generator_foundation
from .keybed_prompts import prompt_generator_keybed_descriptor
from .oneshot_prompts import (
    prompt_generator_oneshot_model_router,
    resolve_oneshot_register_from_ui,
)


# Lightweight compatibility aliases for callers that historically obtained the
# legacy model generators through master_prompt_map.
prompt_generator_piano = piano_prompts.prompt_generator_piano
prompt_generator_edm = edm_elements_prompts.prompt_generator_edm
prompt_generator_vocal_textures = vocal_textures_prompts.prompt_generator_vocal_textures


# Unknown / unclassified loop checkpoints use Foundation as the modern default.
def default_prompt_generator(**kwargs):
    return prompt_generator_foundation(**kwargs)


def is_keybed_capable_model(model_name):
    return bool(re.search(r"keybed", model_name or "", re.IGNORECASE))


def is_oneshot_capable_model(model_name):
    return bool(
        re.search(
            r"(?:one[_\-\s]?shot|oneshot)",
            model_name or "",
            re.IGNORECASE,
        )
    )


def is_foundation_model(model_name):
    """Return True for Foundation loop/sample checkpoints.

    Foundation keybed checkpoints are intentionally excluded here because they
    retain their specialized keybed routing through get_prompt_generator().
    """
    return bool(re.search(r"foundation", model_name or "", re.IGNORECASE)) and not is_keybed_capable_model(model_name)


def get_loop_prompt_family(model_name):
    """Return the loop prompt-builder family for a checkpoint name.

    Legacy model names retain their historical regex routing. Everything else,
    including Foundation-1.2-Samples and unknown checkpoints, falls back to the
    general Foundation prompt system.
    """
    name = model_name or ""

    patterns = (
        (r"piano[s]?.*\.(ckpt|safetensors)$", "piano"),
        (r"edm.*elements.*\.(ckpt|safetensors)$", "edm_elements"),
        (r"vocal.*textures.*\.(ckpt|safetensors)$", "vocal_textures"),
    )

    for pattern, family in patterns:
        if re.search(pattern, name, re.IGNORECASE):
            return family

    return "foundation"


def get_loop_prompt_generator(model_name):
    """Return the loop prompt generator for the detected model family."""
    family = get_loop_prompt_family(model_name)

    if family == "piano":
        return piano_prompts.prompt_generator_piano
    if family == "edm_elements":
        return edm_elements_prompts.prompt_generator_edm
    if family == "vocal_textures":
        return vocal_textures_prompts.prompt_generator_vocal_textures

    return prompt_generator_foundation


def get_prompt_generator(model_name):
    """Compatibility router for model-aware prompt generation.

    Explicit keybed and one-shot checkpoint names retain their specialized
    generators. All regular loop models delegate to get_loop_prompt_generator().
    """
    name = model_name or ""

    if re.search(r"keybed.*\.(ckpt|safetensors)$", name, re.IGNORECASE):
        return prompt_generator_keybed_descriptor

    if re.search(
        r"(?:one[_\-\s]?shot|oneshot).*\.(ckpt|safetensors)$",
        name,
        re.IGNORECASE,
    ):
        return prompt_generator_oneshot_model_router

    return get_loop_prompt_generator(name)
