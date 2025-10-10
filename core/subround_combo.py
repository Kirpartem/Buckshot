from dataclasses import dataclass

@dataclass
class SubroundCombo:
    num_items: int
    starting_hp: int
    blanks: int
    lives: int

# Presets: mostly 6-round cylinders (blanks + lives = 6) for consistency
combos = {
    # ── Gentle / On-ramp ──────────────────────────────────────────────────────
    "training_wheels":  SubroundCombo(1, 3, 3, 1),
    "balanced_basic":   SubroundCombo(2, 3, 3, 3),
    "light_load":       SubroundCombo(0, 3, 4, 2),
    "campfire":         SubroundCombo(1, 3, 4, 2),
    "milk_run":         SubroundCombo(2, 3, 4, 2),
    "airsoft_day":      SubroundCombo(3, 3, 5, 1),

    # ── Standard / Fair fights ────────────────────────────────────────────────
    "even_steven":      SubroundCombo(2, 3, 3, 3),
    "tug_of_war":       SubroundCombo(1, 3, 2, 4),
    "sparring_match":   SubroundCombo(2, 3, 2, 4),
    "coinflip":         SubroundCombo(0, 3, 3, 3),
    "trade_blows":      SubroundCombo(3, 3, 2, 4),
    "bruiser":          SubroundCombo(1, 4, 3, 3),

    # ── Item-forward (toys change the math) ───────────────────────────────────
    "toolbox_round":    SubroundCombo(4, 3, 2, 3),
    "loaded_pockets":   SubroundCombo(3, 4, 3, 2),
    "gadgeteer":        SubroundCombo(4, 3, 3, 3),
    "tinker_time":      SubroundCombo(4, 3, 4, 2),
    "resupply":         SubroundCombo(3, 3, 4, 2),

    # ── Pressure cookers (spicier but not stupid) ────────────────────────────
    "hot_barrel":       SubroundCombo(1, 3, 1, 4),
    "knife_edge":       SubroundCombo(1, 2, 1, 4),
    "last_cigarette":   SubroundCombo(2, 2, 2, 3),
    "red_alert":        SubroundCombo(2, 3, 1, 5),
    "glass_and_pray":   SubroundCombo(3, 2, 2, 4),

    # ── High risk / Punishing (but still playable) ───────────────────────────
    "doom_spin":        SubroundCombo(2, 3, 0, 4),
    "blood_rush":       SubroundCombo(2, 2, 2, 4),
    "sudden_death":     SubroundCombo(0, 2, 1, 3),
    "no_mistakes":      SubroundCombo(1, 2, 2, 4),
    "thin_ice":         SubroundCombo(0, 2, 3, 3),

    # ── Lower tempo / Breathers ──────────────────────────────────────────────
    "empty_chamber":    SubroundCombo(0, 4, 5, 1),
    "coffee_break":     SubroundCombo(1, 4, 4, 2),
    "second_wind":      SubroundCombo(2, 4, 3, 3),
    "catch_breath":     SubroundCombo(2, 3, 5, 1),

    # ── Quirky balances / Oddballs ───────────────────────────────────────────
    "gambler_choice":   SubroundCombo(4, 3, 1, 3),
    "ironman":          SubroundCombo(1, 4, 2, 4),
    "bare_knuckle":     SubroundCombo(0, 3, 3, 2),
    "inventory_overflow": SubroundCombo(4, 3, 4, 1),
    "saw_fest":         SubroundCombo(3, 3, 2, 4),
    "cuff_n_load":      SubroundCombo(3, 3, 2, 2),

    # ── Control freak sets (knowledge and tempo matter) ──────────────────────
    "peekaboo":         SubroundCombo(3, 3, 3, 3),
    "lock_and_key":     SubroundCombo(3, 3, 4, 2),
    "tactical_pause":   SubroundCombo(2, 3, 5, 1),
    "tempo_thief":      SubroundCombo(4, 3, 3, 3),

    # ── Swingy but fair (momentum flips possible) ────────────────────────────
    "momentum":         SubroundCombo(2, 3, 2, 4),
    "pendulum":         SubroundCombo(1, 3, 4, 2),
    "seesaw":           SubroundCombo(2, 3, 4, 2),
    "teeter":           SubroundCombo(1, 3, 3, 3),

    # ── Low-item skill checks ────────────────────────────────────────────────
    "bare_minimum":     SubroundCombo(0, 3, 4, 2),
    "steel_nerves":     SubroundCombo(0, 3, 2, 4),
    "no_training_wheels": SubroundCombo(0, 3, 3, 3),

    # ── High-item playgrounds ────────────────────────────────────────────────
    "pocket_santa":     SubroundCombo(4, 3, 2, 4),
    "kit_bash":         SubroundCombo(4, 3, 3, 3),
    "trick_bag":        SubroundCombo(4, 3, 4, 2),

    # ── HP variance samplers (still within 2–4) ─────────────────────────────
    "glass_cannon":     SubroundCombo(2, 2, 3, 3),
    "stout_heart":      SubroundCombo(2, 4, 2, 4),
    "thick_skin":       SubroundCombo(1, 4, 3, 3),
    "fragile_speedrun": SubroundCombo(1, 2, 3, 3),

    # ── Thematic six-shooters (exactly 6 total) ─────────────────────────────
    "classic_even":     SubroundCombo(2, 3, 3, 3),
    "tilt_blanks":      SubroundCombo(2, 3, 4, 2),
    "tilt_live":        SubroundCombo(2, 3, 2, 4),
    "edge_live":        SubroundCombo(1, 3, 1, 5),
    "edge_blank":       SubroundCombo(1, 3, 5, 1),
}
