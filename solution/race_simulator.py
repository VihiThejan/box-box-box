#!/usr/bin/env python3
import json
import os
import sys
from typing import Dict, List, Tuple


TIRES = ("SOFT", "MEDIUM", "HARD")
KNOTS = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)
TRACKS = ("Bahrain", "COTA", "Monaco", "Monza", "Silverstone", "Spa", "Suzuka")


FEATURE_NAMES = [
    "pit_count",
    "pit_lane_scaled",
    "start_soft",
    "start_medium",
    "start_hard",
    "compound_count",
    "invalid_compound_rule",
    "base_lap_total",
    "laps_SOFT",
    "sum_age_SOFT",
    "sum_age2_SOFT",
    "sum_age3_SOFT",
    "stints_SOFT",
    "temp_age_SOFT",
    "temp_age2_SOFT",
    "laps_MEDIUM",
    "sum_age_MEDIUM",
    "sum_age2_MEDIUM",
    "sum_age3_MEDIUM",
    "stints_MEDIUM",
    "temp_age_MEDIUM",
    "temp_age2_MEDIUM",
    "laps_HARD",
    "sum_age_HARD",
    "sum_age2_HARD",
    "sum_age3_HARD",
    "stints_HARD",
    "temp_age_HARD",
    "temp_age2_HARD",
]


# Learned with solution/train_rank_model.py on historical races.
DEFAULT_WEIGHTS = [
    1383.8483860328033,
    30488.47443579019,
    -7568.009826123202,
    3604.152536079202,
    3963.857290043938,
    593.0144790313315,
    0.0,
    0.0,
    -104987.91001801603,
    -539434.588482819,
    -2984364.4203673033,
    291863.2773555751,
    -11388.923180847529,
    -43646.94421076847,
    36580.67090529309,
    56372.804964005845,
    111194.03835384238,
    -1170230.4333083879,
    58404.69293164841,
    8272.07330720653,
    68530.35039131813,
    13785.37267266649,
    48615.105054037216,
    195944.35546301847,
    -396570.67476850934,
    13652.710361589123,
    4500.698259673661,
    39666.16718993054,
    5898.947080238807,
]


def load_model() -> Tuple[List[str], List[float]]:
    model_path = os.path.join("solution", "model_weights.json")
    if not os.path.exists(model_path):
        return FEATURE_NAMES, DEFAULT_WEIGHTS
    try:
        with open(model_path, "r", encoding="utf-8") as f:
            model = json.load(f)
        names = model.get("feature_names")
        weights = model.get("weights")
        if isinstance(names, list) and isinstance(weights, list) and len(names) == len(weights) and len(names) > 0:
            return names, [float(w) for w in weights]
    except Exception:
        pass
    return FEATURE_NAMES, DEFAULT_WEIGHTS


MODEL_FEATURES, MODEL_WEIGHTS = load_model()


LINEAR_SCALE = 1_000_000.0
PHYSICS_WEIGHT = 0.0


PHYS_TIRE_BASE = {"SOFT": -0.82, "MEDIUM": 0.0, "HARD": 0.94}
PHYS_TEMP_COEFF = {"SOFT": 0.0018, "MEDIUM": 0.0012, "HARD": 0.0009}
PHYS_DEG_LIN = {"SOFT": 0.0175, "MEDIUM": 0.0120, "HARD": 0.0082}
PHYS_DEG_QUAD = {"SOFT": 0.00046, "MEDIUM": 0.00028, "HARD": 0.00016}
PHYS_KNEE = {"SOFT": 10, "MEDIUM": 13, "HARD": 16}


def build_stints(total_laps: int, starting_tire: str, pit_stops: List[dict]) -> List[Tuple[str, int]]:
    pit_map = {int(stop["lap"]): stop["to_tire"] for stop in pit_stops}
    stints: List[Tuple[str, int]] = []
    current_tire = starting_tire
    current_len = 0
    for lap in range(1, total_laps + 1):
        current_len += 1
        if lap in pit_map:
            stints.append((current_tire, current_len))
            current_tire = pit_map[lap]
            current_len = 0
    if current_len > 0:
        stints.append((current_tire, current_len))
    return stints


def build_feature_map(race_config: dict, strategy: dict) -> Dict[str, float]:
    total_laps = int(race_config["total_laps"])
    pit_lane_time = float(race_config["pit_lane_time"])
    temp_delta = float(race_config["track_temp"]) - 30.0
    base_lap = float(race_config["base_lap_time"])

    stints = build_stints(total_laps, strategy["starting_tire"], strategy.get("pit_stops", []))
    pits = len(strategy.get("pit_stops", []))

    feats: Dict[str, float] = {
        "pit_count": float(pits),
        "pit_lane_scaled": float(pits) * pit_lane_time,
        "start_soft": 1.0 if strategy["starting_tire"] == "SOFT" else 0.0,
        "start_medium": 1.0 if strategy["starting_tire"] == "MEDIUM" else 0.0,
        "start_hard": 1.0 if strategy["starting_tire"] == "HARD" else 0.0,
        "compound_count": 0.0,
        "invalid_compound_rule": 0.0,
        "base_lap_total": base_lap * total_laps,
    }

    driver_id = strategy["driver_id"]
    track = race_config["track"]
    for i in range(1, 21):
        d = f"D{i:03d}"
        feats[f"driver_{d}"] = 1.0 if d == driver_id else 0.0
    for t in TRACKS:
        feats[f"track_{t}"] = 1.0 if t == track else 0.0

    used_compounds = set()
    for tire in TIRES:
        feats[f"laps_{tire}"] = 0.0
        feats[f"sum_age_{tire}"] = 0.0
        feats[f"sum_age2_{tire}"] = 0.0
        feats[f"sum_age3_{tire}"] = 0.0
        feats[f"stints_{tire}"] = 0.0
        feats[f"temp_age_{tire}"] = 0.0
        feats[f"temp_age2_{tire}"] = 0.0
        for k in KNOTS:
            feats[f"hinge1_{tire}_{k}"] = 0.0
            feats[f"hinge2_{tire}_{k}"] = 0.0
            feats[f"temp_hinge1_{tire}_{k}"] = 0.0
            feats[f"temp_hinge2_{tire}_{k}"] = 0.0

    for tire, length in stints:
        used_compounds.add(tire)
        s1 = length * (length + 1) / 2.0
        s2 = length * (length + 1) * (2 * length + 1) / 6.0
        s3 = s1 * s1
        feats[f"laps_{tire}"] += float(length)
        feats[f"sum_age_{tire}"] += s1
        feats[f"sum_age2_{tire}"] += s2
        feats[f"sum_age3_{tire}"] += s3
        feats[f"stints_{tire}"] += 1.0
        feats[f"temp_age_{tire}"] += temp_delta * s1
        feats[f"temp_age2_{tire}"] += temp_delta * s2
        for k in KNOTS:
            if length > k:
                n = length - k
                h1 = n * (n + 1) / 2.0
                h2 = n * (n + 1) * (2 * n + 1) / 6.0
                feats[f"hinge1_{tire}_{k}"] += h1
                feats[f"hinge2_{tire}_{k}"] += h2
                feats[f"temp_hinge1_{tire}_{k}"] += temp_delta * h1
                feats[f"temp_hinge2_{tire}_{k}"] += temp_delta * h2

    feats["compound_count"] = float(len(used_compounds))
    feats["invalid_compound_rule"] = 1.0 if len(used_compounds) < 2 else 0.0
    for t in TRACKS:
        tr = feats[f"track_{t}"]
        for tire in TIRES:
            feats[f"track_{t}_sum_age_{tire}"] = tr * feats[f"sum_age_{tire}"]
            feats[f"track_{t}_sum_age2_{tire}"] = tr * feats[f"sum_age2_{tire}"]
    return feats


def predict_driver_score(race_config: dict, strategy: dict) -> float:
    feats = build_feature_map(race_config, strategy)
    linear = 0.0
    for name, w in zip(MODEL_FEATURES, MODEL_WEIGHTS):
        linear += w * feats.get(name, 0.0)

    if PHYSICS_WEIGHT == 0.0:
        return linear / LINEAR_SCALE

    total_laps = int(race_config["total_laps"])
    base_lap = float(race_config["base_lap_time"])
    track_temp = float(race_config["track_temp"])
    pit_lane_time = float(race_config["pit_lane_time"])
    pit_map = {int(stop["lap"]): stop["to_tire"] for stop in strategy.get("pit_stops", [])}
    tire = strategy["starting_tire"]
    age = 0
    physics = 0.0
    temp_delta = track_temp - 30.0

    for lap in range(1, total_laps + 1):
        age += 1
        d = PHYS_DEG_LIN[tire] * age
        if age > PHYS_KNEE[tire]:
            x = age - PHYS_KNEE[tire]
            d += PHYS_DEG_QUAD[tire] * x * x
        physics += base_lap + PHYS_TIRE_BASE[tire] + PHYS_TEMP_COEFF[tire] * temp_delta * age + d
        if lap in pit_map:
            physics += pit_lane_time
            tire = pit_map[lap]
            age = 0

    return linear / LINEAR_SCALE + PHYSICS_WEIGHT * physics


def simulate_race(race_input: dict) -> List[str]:
    race_config = race_input["race_config"]
    strategies = race_input["strategies"]

    totals: List[Tuple[str, float]] = []
    for pos_key, strategy in strategies.items():
        _ = pos_key
        driver_id = strategy["driver_id"]
        score = predict_driver_score(race_config, strategy)
        totals.append((driver_id, score))

    totals.sort(key=lambda x: x[1])
    return [driver_id for driver_id, _ in totals]


def simulate_race_from_parts(
    race_config: dict,
    strategies: dict,
) -> List[str]:
    return simulate_race({"race_config": race_config, "strategies": strategies})


def main() -> None:
    race_input = json.load(sys.stdin)
    output = {
        "race_id": race_input["race_id"],
        "finishing_positions": simulate_race(race_input),
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()