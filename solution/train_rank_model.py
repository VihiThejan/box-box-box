#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import random
from typing import Dict, List, Tuple


TIRES = ("SOFT", "MEDIUM", "HARD")
KNOTS = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)
TRACKS = ("Bahrain", "COTA", "Monaco", "Monza", "Silverstone", "Spa", "Suzuka")


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


def features_for_driver(race_config: dict, strategy: dict) -> Dict[str, float]:
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


def vec_dot(w: List[float], x: List[float]) -> float:
    s = 0.0
    for i in range(len(w)):
        s += w[i] * x[i]
    return s


def add_scaled(dst: List[float], src: List[float], scale: float) -> None:
    for i in range(len(dst)):
        dst[i] += src[i] * scale


def sigmoid(v: float) -> float:
    if v >= 0:
        z = math.exp(-v)
        return 1.0 / (1.0 + z)
    z = math.exp(v)
    return z / (1.0 + z)


def load_races(data_dir: str, max_races: int) -> List[dict]:
    races: List[dict] = []
    for path in sorted(glob.glob(os.path.join(data_dir, "races_*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            races.extend(json.load(f))
        if len(races) >= max_races:
            break
    return races[:max_races]


def build_dataset(races: List[dict], feature_names: List[str]) -> List[dict]:
    data = []
    for race in races:
        by_driver = {}
        for strategy in race["strategies"].values():
            feats = features_for_driver(race["race_config"], strategy)
            by_driver[strategy["driver_id"]] = [feats.get(name, 0.0) for name in feature_names]
        data.append({
            "x": by_driver,
            "order": race["finishing_positions"],
        })
    return data


def eval_dataset(data: List[dict], w: List[float]) -> Tuple[float, float]:
    exact = 0
    pair = 0.0
    for race in data:
        order = race["order"]
        scores = {d: vec_dot(w, x) for d, x in race["x"].items()}
        pred = sorted(scores.keys(), key=lambda d: scores[d])
        if pred == order:
            exact += 1

        pos = {d: i for i, d in enumerate(pred)}
        ok = 0
        total = 0
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                total += 1
                if pos[order[i]] < pos[order[j]]:
                    ok += 1
        pair += ok / total
    n = len(data)
    return exact / n, pair / n


def train(data_train: List[dict], data_val: List[dict], feature_names: List[str], epochs: int, seed: int) -> List[float]:
    rng = random.Random(seed)
    w = [0.0 for _ in feature_names]
    lr0 = 0.03
    l2 = 1e-6

    best_w = list(w)
    best_score = -1.0

    for epoch in range(1, epochs + 1):
        rng.shuffle(data_train)
        lr = lr0 / (1.0 + 0.06 * epoch)

        for race in data_train:
            order = race["order"]
            x = race["x"]

            for _ in range(45):
                i = rng.randint(0, len(order) - 2)
                j = rng.randint(i + 1, len(order) - 1)
                d_fast = order[i]
                d_slow = order[j]
                xf = x[d_fast]
                xs = x[d_slow]

                diff = vec_dot(w, xf) - vec_dot(w, xs)
                g = sigmoid(diff)
                for k in range(len(w)):
                    grad = g * (xf[k] - xs[k]) + l2 * w[k]
                    w[k] -= lr * grad

        train_exact, train_pair = eval_dataset(data_train[: min(250, len(data_train))], w)
        val_exact, val_pair = eval_dataset(data_val, w)
        score = val_exact * 10.0 + val_pair
        if score > best_score:
            best_score = score
            best_w = list(w)
            print(
                f"epoch={epoch:03d} train_exact={train_exact:.4f} train_pair={train_pair:.4f} "
                f"val_exact={val_exact:.4f} val_pair={val_pair:.4f}"
            )

    return best_w


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a linear pairwise ranking model")
    parser.add_argument("--data-dir", default="data/historical_races")
    parser.add_argument("--races", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--model-out", default="solution/model_weights.json")
    args = parser.parse_args()

    feature_names = [
        "pit_count",
        "pit_lane_scaled",
        "start_soft",
        "start_medium",
        "start_hard",
        "compound_count",
        "invalid_compound_rule",
        "base_lap_total",
    ]
    for i in range(1, 21):
        feature_names.append(f"driver_D{i:03d}")
    for t in TRACKS:
        feature_names.append(f"track_{t}")
    for tire in TIRES:
        feature_names.extend(
            [
                f"laps_{tire}",
                f"sum_age_{tire}",
                f"sum_age2_{tire}",
                f"sum_age3_{tire}",
                f"stints_{tire}",
                f"temp_age_{tire}",
                f"temp_age2_{tire}",
            ]
        )
        for k in KNOTS:
            feature_names.extend(
                [
                    f"hinge1_{tire}_{k}",
                    f"hinge2_{tire}_{k}",
                    f"temp_hinge1_{tire}_{k}",
                    f"temp_hinge2_{tire}_{k}",
                ]
            )
    for t in TRACKS:
        for tire in TIRES:
            feature_names.append(f"track_{t}_sum_age_{tire}")
            feature_names.append(f"track_{t}_sum_age2_{tire}")

    races = load_races(args.data_dir, args.races)
    split = int(len(races) * 0.9)
    train_races = races[:split]
    val_races = races[split:]

    data_train = build_dataset(train_races, feature_names)
    data_val = build_dataset(val_races, feature_names)

    weights = train(data_train, data_val, feature_names, args.epochs, args.seed)
    val_exact, val_pair = eval_dataset(data_val, weights)
    print("BEST_WEIGHTS")
    model = {"feature_names": feature_names, "weights": weights}
    print(json.dumps(model))
    print(f"FINAL val_exact={val_exact:.4f} val_pair={val_pair:.4f}")

    with open(args.model_out, "w", encoding="utf-8") as f:
        json.dump(model, f)
    print(f"MODEL_SAVED {args.model_out}")


if __name__ == "__main__":
    main()