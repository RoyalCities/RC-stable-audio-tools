"""Legacy Vocal Textures prompt vocabulary and builder helpers."""
import random

VOCAL_TYPES = ["Male Vocal Texture", "Female Vocal Texture", "Ensemble Vocal Texture"]
STRUCTURE_CHOICES = ["chord progression"]


def build_prompt(vocal_type=None, structure=None):
    vocal = str(vocal_type or VOCAL_TYPES[0]).strip()
    structure = str(structure or STRUCTURE_CHOICES[0]).strip()
    return f"{vocal}, {structure}"


def prompt_generator_vocal_textures(*, return_plan=False, **_):
    vocal = random.choice(VOCAL_TYPES)
    structure = STRUCTURE_CHOICES[0]
    prompt = build_prompt(vocal, structure)
    if return_plan:
        return {"prompt": prompt, "vocal_type": vocal, "structure": structure}
    return prompt

prompt_generator = prompt_generator_vocal_textures
