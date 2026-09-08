"""Legacy RC Infinite Pianos prompt vocabulary and builder helpers."""
import random

PIANO_TYPES = ["Soft E. Piano", "Medium E. Piano", "Grand Piano"]
TREMOLO_EFFECTS = ["Low Tremolo", "Medium Tremolo", "High Tremolo", "No Tremolo"]
NON_TREMOLO_EFFECTS = ["No Reverb", "Low Reverb", "Medium Reverb", "High Reverb", "High Spacey Reverb"]
PIANO_EFFECT_CHOICES = NON_TREMOLO_EFFECTS + TREMOLO_EFFECTS

CHORD_STYLES = [
    "simple", "complex", "dance plucky", "fast", "jazzy", "low", "simple strummed",
    "rising strummed", "complex strummed", "jazzy strummed", "slow strummed", "plucky dance",
    "rising", "falling", "slow", "slow jazzy", "fast jazzy", "smooth", "strummed", "plucky",
]
MELODY_STYLES = [
    "catchy melody", "complex melody", "complex top melody", "catchy top melody", "top melody",
    "smooth melody", "catchy complex melody", "jazzy melody", "smooth catchy melody",
    "plucky dance melody", "dance melody", "alternating low melody", "alternating top arp melody",
    "alternating top melody", "top arp melody", "alternating melody", "falling arp melody",
    "rising arp melody", "top catchy melody",
]
STRUCTURE_CHOICES = ["Chord Progression Only", "Chord Progression + Melody", "Melody Only"]
NONE = "None"


def effect_choices_for_piano(piano_type):
    if str(piano_type or "").casefold() == "grand piano":
        return list(NON_TREMOLO_EFFECTS)
    return list(NON_TREMOLO_EFFECTS + TREMOLO_EFFECTS)


def build_prompt(piano_type=None, structure=None, chord_style=None, melody_style=None, effect=None):
    piano = str(piano_type or PIANO_TYPES[0]).strip()
    structure = str(structure or STRUCTURE_CHOICES[0]).strip()
    chord_style = "" if str(chord_style or NONE).casefold() == NONE.casefold() else str(chord_style).strip()
    melody_style = "" if str(melody_style or NONE).casefold() == NONE.casefold() else str(melody_style).strip()
    allowed_effects = effect_choices_for_piano(piano)
    effect = str(effect or allowed_effects[0]).strip()
    if effect not in allowed_effects:
        effect = allowed_effects[0]

    tokens = [piano]
    if structure == "Melody Only":
        tokens.append(f"{melody_style or 'melody'} only")
    elif structure == "Chord Progression + Melody":
        tokens.append(f"{chord_style + ' ' if chord_style else ''}chord progression")
        tokens.append(f"with {melody_style or 'melody'}")
    else:
        tokens.append(f"{chord_style + ' ' if chord_style else ''}chord progression only")
    if effect:
        tokens.append(effect)
    return ", ".join(tokens)


def prompt_generator_piano(*, return_plan=False, **_):
    piano = random.choice(PIANO_TYPES)
    effect = random.choice(NON_TREMOLO_EFFECTS if piano == "Grand Piano" else TREMOLO_EFFECTS + NON_TREMOLO_EFFECTS)
    structure = random.choice(STRUCTURE_CHOICES)
    chord_style = random.choice(CHORD_STYLES)
    melody_style = random.choice(MELODY_STYLES)
    prompt = build_prompt(piano, structure, chord_style, melody_style, effect)
    if return_plan:
        return {
            "prompt": prompt, "piano_type": piano, "structure": structure,
            "chord_style": chord_style, "melody_style": melody_style, "effect": effect,
        }
    return prompt

prompt_generator = prompt_generator_piano
