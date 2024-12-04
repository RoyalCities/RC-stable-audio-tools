
import random
import re

def prompt_generator_piano():
    piano_types = ["Soft E. Piano", "Medium E. Piano", "Grand Piano"]
    tremolo_effects = ["Low Tremolo", "Medium Tremolo", "High Tremolo", "No Tremolo"]
    non_tremolo_effects = ["No Reverb", "Low Reverb", "Medium Reverb", "High Reverb", "High Spacey Reverb"]

    chord_progressions = ["simple", "complex", "dance plucky", "fast", "jazzy", "low", "simple strummed", "rising strummed", "complex strummed", "jazzy strummed", "slow strummed", "plucky dance",
                          "rising", "falling", "slow", "slow jazzy", "fast jazzy", "smooth", "strummed", "plucky"]
    melodies = [
        "catchy melody", "complex melody", "complex top melody", "catchy top melody", "top melody", "smooth melody", "catchy complex melody",
        "jazzy melody", "smooth catchy melody", "plucky dance melody", "dance melody", "alternating low melody", "alternating top arp melody", "alternating top melody", "top arp melody", "alternating melody", "falling arp melody",
        "rising arp melody", "top catchy melody"
    ]

    # Choose the piano type first to ensure an even split
    piano = random.choice(piano_types)

    # Choose effect based on piano type
    if piano == "Grand Piano":
        effect = random.choice(non_tremolo_effects)
    else:
        effect = random.choice(tremolo_effects + non_tremolo_effects)

    # Decide category for generation
    category_choice = random.choice(["chord progression only", "chord progression with melody", "melody only"])
    
    if category_choice == "chord progression only":
        chord_progression = random.choice(chord_progressions) + " chord progression only,"
        descriptor = f"{piano}, {chord_progression} {effect}"
    elif category_choice == "chord progression with melody":
        chord_progression = random.choice(chord_progressions) + " chord progression,"
        melody = "with " + random.choice(melodies) + ","
        descriptor = f"{piano}, {chord_progression} {melody} {effect}"
    else:
        melody = random.choice(melodies) + " only,"
        descriptor = f"{piano}, {melody} {effect}"

    return descriptor

def prompt_generator_edm():
    # Note: Key signatures are handled in the UI code and should not be included in the prompt descriptors.

    # ---------------------------
    # 1. Define Descriptor Categories
    # ---------------------------

    # Polyphonic Presets
    poly_presets = [
        ['Pluck', 'Sine', 'Bright', 'Clean', 'Bell'],
        ['Pluck', 'Sine', 'Bell'],
        ['Saw', 'Synth', 'Warm'], 
        ['Lead', 'Saw', 'Synth', 'Warm', 'Supersaw']
        # Add more polyphonic presets as needed
    ]

    # Monophonic Presets
    mono_presets = [
        ['Lead', 'Square', 'Synth', 'Buzzy', 'Legato'],    # Preset 1
        ['Lead', 'Square', 'Clean', 'Warm'],               # Preset 2
        ['Bass', 'Punchy', 'Sub']                          # Bass Preset
        # Add more monophonic presets as needed
    ]

    # Corresponding Weights for Mono Presets
    mono_weights = [30, 30, 40]  # Preset 1: 30%, Preset 2: 30%, Bass Preset: 40%

    # Arpeggio Descriptors
    arpeggio_prompts = [
        "medium speed, alternating arp",
        "medium speed, alternating arp, triplets",
        "fast speed, alternating arp",
        "fast speed, alternating arp, triplets",
        "medium speed, alternating arp melody",
        "medium speed, alternating arp melody, triplets",
        "fast speed, alternating arp melody",
        "fast speed, alternating arp melody, triplets",
        "alternating arp",
        "slow simple melody",
        "rising melody",
        "repeating, simple melody",
        "repeating, catchy melody",
        "repeating catchy, bounce melody",
        "repeating, bounce, catchy, melody",
        "catchy, bounce, melody",
        "catchy, triplets, bounce melody",
        "bounce, top, catchy melody",
        "slow rising arp",
        "slow alternating arp",
        "slow falling arp",
        "arp chord progression",
        "arp melody",
        "arp melody, triplets",
        "arp rising melody",
        "arp catchy melody",
        "arp catchy melody, triplets",
        "alternating arp, triplets",
        "alternating arp melody",
        "alternating arp melody, triplets",
        "fast speed, rising arp melody",
        "fast speed, rising arp melody, triplets",
        "medium speed, rising arp melody",
        "medium speed, rising arp melody, triplets",
        "fast speed falling arp melody",
        "fast speed falling arp melody, triplets",
        "medium speed, falling arp melody",
        "catchy, top simple melody",
        "catchy, repeating, off beat melody",
        "medium speed, falling arp melody, triplets",
        "simple, off beat, catchy melody"
        # Add more arpeggio descriptors as needed
    ]

    # Chord Progressions
    chord_progressions = [
        "dance", "complex", "", "catchy dance", "fast speed", "medium speed", "fast speed, strummed",
        "medium speed, strummed", "pluck", "rising dance",
        "simple dance", "slow strummed", "slow speed"
        # Add more chord progressions as needed
    ]

    # Melodies
    melodies = [
        "alternating arp", "alternating catchy melody",
        "alternating arp melody", "catchy melody", 
        "melody", "off beat simple catchy melody", "repeating catchy melody",
        "repeating melody",
        "simple alternating arp melody", "simple catchy melody", "simple falling melody", "simple melody", 
        "simple slow melody", "simple off beat catchy melody", "slow top melody",
        "top catchy melody", "top slow melody", "top repeating catchy melody", "top slow melody", 
        # Add more melodies as needed
    ]

    # New Specific Chord Progression Descriptors
    chord_progression_specific_prompts = [
        "chord progression with catchy melody",
        "chord progression with catchy repeating melody",
        "chord progression with complex melody",
        "chord progression with melody",
        "chord progression with off beat simple catchy melody",
        "chord progression with repeating catchy melody",
        "chord progression with repeating melody",
        "complex chord progression with melody",
        "complex chord progression with top simple melody",
        "dance chord progression",
        "dance chord progression with catchy melody",
        "dance chord progression with complex dance melody",
        "dance chord progression with complex rising arp",
        "dance chord progression with dance catchy melody",
        "dance chord progression with off beat",
        "dance chord progression with off beat catchy melody",
        "dance chord progression with off beat melody",
        "dance chord progression with off beat simple melody",
        "dance chord progression with off beat top melody",
        "dance chord progression with rising arp melody",
        "dance chord progression with simple catchy melody",
        "dance chord progression with simple melody",
        "dance chord progression with simple top catchy melody",
        "dance chord progression with slow beat simple melody",
        "dance chord progression with top",
        "dance chord progression with top catchy melody",
        "dance chord progression with top catchy repeating melody",
        "dance chord progression with top dance melody",
        "dance chord progression with top melody",
        "dance chord progression with top repeating melody",
        "dance chord progression with top simple melody",
        "dance chord progression with top slow melody",
        "dance progression",
        "dance progression with simple catchy melody",
        "dance progression with top melody",
        "dance simple chord progression",
        "dance slow chord progression",
        "medium speed chord progression with top catchy melody",
        "pluck chord progression with top alternating melody",
        "pluck chord progression with top catchy melody",
        "pluck chord progression with top slow melody",
        "rising dance chord progression",
        "rising dance chord progression with off beat repeating melody",
        "rising dance chord progression with top catchy melody",
        "simple dance chord progression",
        "simple dance chord progression with alternating arp melody",
        "simple dance chord progression with simple melody",
        "simple dance chord progression with simple off beat melody",
        "simple dance chord progression with simple top melody",
        "simple dance chord progression with slow top melody",
        "simple dance chord progression with top dance melody",
        "simple dance chord progression with top melody"
    ]

    # ---------------------------
    # 3. Define Prompt Categories and Probabilities
    # ---------------------------

    prompt_categories = [
        "arpeggio_only",
        "chord_progression_only",
        "chord_progression_with_melody",
        "chord_progression_specific",   # New Category Added
        "melody_only"
    ]

    # Probability Percentages for selecting prompt categories
    prompt_probabilities = [20, 15, 25, 20, 20]  # Sum = 100

    # ---------------------------
    # 4. Decide Prompt Type
    # ---------------------------
    prompt_type = random.choices(
        prompt_categories,
        weights=prompt_probabilities,
        k=1
    )[0]

    initial_descriptors = []
    specific_descriptors = []

    # ---------------------------
    # 5. Generate Descriptors Based on Prompt Type
    # ---------------------------

    if prompt_type == "arpeggio_only":
        # Arpeggio Only: Must include exactly three descriptors from either polyphonic or monophonic presets

        # Decide whether to use polyphonic or monophonic presets
        # 50% polyphonic, 50% monophonic
        preset_type = random.choices(
            ['polyphonic', 'monophonic'],
            weights=[50, 50],
            k=1
        )[0]

        if preset_type == 'polyphonic':
            preset = random.choice(poly_presets)
        else:
            # Use weighted selection for mono presets
            preset = random.choices(
                mono_presets,
                weights=mono_weights,
                k=1
            )[0]

        # Select exactly three descriptors from the chosen preset
        selected_descriptors = random.sample(preset, min(3, len(preset)))
        initial_descriptors.extend(selected_descriptors)

        # Add an arpeggio descriptor
        arpeggio = random.choice(arpeggio_prompts)
        specific_descriptors.append(arpeggio)

    elif prompt_type == "chord_progression_only":
        # Chord Progression Only: Only polyphonic presets

        # Select a polyphonic preset
        preset = random.choice(poly_presets)

        # Select exactly three descriptors from the chosen preset
        selected_descriptors = random.sample(preset, min(3, len(preset)))
        initial_descriptors.extend(selected_descriptors)

        # Add a chord progression descriptor
        chord_prog = random.choice(chord_progressions)
        if chord_prog:
            chord_prog += " chord progression"
        else:
            chord_prog = "chord progression"
        specific_descriptors.append(chord_prog)

    elif prompt_type == "chord_progression_with_melody":
        # Chord Progression with Melodies: Only polyphonic presets

        # Select a polyphonic preset
        preset = random.choice(poly_presets)

        # Select exactly three descriptors from the chosen preset
        selected_descriptors = random.sample(preset, min(3, len(preset)))
        initial_descriptors.extend(selected_descriptors)

        # Add a chord progression descriptor
        chord_prog = random.choice(chord_progressions)
        if chord_prog:
            chord_prog += " chord progression"
        else:
            chord_prog = "chord progression"
        specific_descriptors.append(chord_prog)

        # Add a melody descriptor
        melody = random.choice(melodies)
        specific_descriptors.append(f"with {melody}")

    elif prompt_type == "chord_progression_specific":
        # Chord Progression Specific: Only polyphonic presets and use specific chord progression descriptors

        # Select a polyphonic preset
        preset = random.choice(poly_presets)

        # Select exactly three descriptors from the chosen preset
        selected_descriptors = random.sample(preset, min(3, len(preset)))
        initial_descriptors.extend(selected_descriptors)

        # Add a specific chord progression descriptor
        chord_prog_specific = random.choice(chord_progression_specific_prompts)
        specific_descriptors.append(chord_prog_specific)

    elif prompt_type == "melody_only":
        # Melody Only: Must include exactly three descriptors from either polyphonic or monophonic presets

        # Decide whether to use polyphonic or monophonic presets
        # 60% polyphonic, 40% monophonic
        preset_type = random.choices(
            ['polyphonic', 'monophonic'],
            weights=[60, 40],
            k=1
        )[0]

        if preset_type == 'polyphonic':
            preset = random.choice(poly_presets)
        else:
            # Use weighted selection for mono presets
            preset = random.choices(
                mono_presets,
                weights=mono_weights,
                k=1
            )[0]

        # Select exactly three descriptors from the chosen preset
        selected_descriptors = random.sample(preset, min(3, len(preset)))
        initial_descriptors.extend(selected_descriptors)

        # Add a melody descriptor
        melody = random.choice(melodies) + " only"
        specific_descriptors.append(melody)

    # ---------------------------
    # 7. Handle Effects with Probabilities
    # ---------------------------
    effects = []

    # Decide how many effect categories to apply (0 to 3)
    # Probabilities: 0 effects (10%), 1 effect (60%), 2 effects (25%), 3 effects (5%)
    num_effect_categories = random.choices(
        [0, 1, 2, 3],
        weights=[10, 60, 25, 5],
        k=1
    )[0]

    if num_effect_categories > 0:
        # Define effects categories
        effect_categories = {
            'reverb': ["small reverb", "medium reverb", "high reverb"],
            'filter_sweep': ["with falling high-cut", "with rising low-pass"],
            'gate': ["with half-beat gate", "with quarter-beat gate"]
        }

        # Define weights for effect categories
        effect_category_weights = {
            'reverb': 45,
            'filter_sweep': 45,
            'gate': 10
        }

        # Create a list of categories weighted by their probabilities
        categories = list(effect_categories.keys())
        weights = [effect_category_weights[cat] for cat in categories]

        # To select multiple categories without replacement, perform weighted sampling manually
        selected_categories = []
        available_categories = categories.copy()
        available_weights = weights.copy()

        for _ in range(num_effect_categories):
            if not available_categories:
                break
            chosen = random.choices(
                available_categories,
                weights=available_weights,
                k=1
            )[0]
            selected_categories.append(chosen)
            # Remove the chosen category to avoid duplication
            index = available_categories.index(chosen)
            del available_categories[index]
            del available_weights[index]

        # Now, select one effect from each chosen category
        for category in selected_categories:
            effect = random.choice(effect_categories[category])
            effects.append(effect)

    # ---------------------------
    # 8. Assemble the Descriptor
    # ---------------------------
    # Ensure that initial descriptors are first, followed by specific descriptors
    descriptor = ", ".join(initial_descriptors + specific_descriptors)

    if effects:
        descriptor += ", " + ", ".join(effects)

    return descriptor

def prompt_generator_vocal_textures():
    vocal_types = ["Male Vocal Texture", "Female Vocal Texture", "Ensemble Vocal Texture"]
    vocal = random.choice(vocal_types)
    descriptor = f"{vocal}, chord progression,"
    return descriptor

def default_prompt_generator():

    # Generic descriptors
    descriptors = [
        "arp",
        "chord progression",
        "catchy melody",
        "chord progression with top melody",
        "top melody"        
    ]

    descriptor = random.choice(descriptors)
    return descriptor

def get_prompt_generator(model_name):
    # Adjusted regex patterns and using re.search
    patterns = [
        (r'piano[s]?.*\.ckpt', prompt_generator_piano),
        (r'edm.*elements.*\.ckpt', prompt_generator_edm),
        (r'vocal.*textures.*\.ckpt', prompt_generator_vocal_textures),
        # Add more patterns if necessary
    ]

    for pattern, generator in patterns:
        if re.search(pattern, model_name, re.IGNORECASE):
            return generator

    # If no pattern matches, return the default prompt generator
    return default_prompt_generator