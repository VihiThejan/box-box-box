#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


TIRES = ("SOFT", "MEDIUM", "HARD")


@dataclass(frozen=True)
class ModelParams:
    tire_base: Dict[str, float]
    temp_coeff: Dict[str, float]
    deg_lin: Dict[str, float]
    deg_quad: Dict[str, float]
    knee: Dict[str, int]


def build_pit_map(pit_stops: List[dict]) -> Dict[int, str]:
    pit_map: Dict[int, str] = {}
    for stop in pit_stops:
        pit_map[int(stop["lap"])] = stop["to_tire"]
    return pit_map


def degradation_for_age(age: int, tire: str, params: ModelParams) -> float:
    knee = params.knee[tire]
    if age <= knee:
        return params.deg_lin[tire] * age
    d = age - knee
    return params.deg_lin[tire] * age + params.deg_quad[tire] * d * d


def simulate_driver_total_time(race_config: dict, strategy: dict, params: ModelParams) -> float:
    total_laps = int(race_config["total_laps"])
    base_lap = float(race_config["base_lap_time"])
    track_temp = float(race_config["track_temp"])
    pit_lane_time = float(race_config["pit_lane_time"])

    tire = strategy["starting_tire"]
    pit_map = build_pit_map(strategy.get("pit_stops", []))

    tire_age = 0
    total_time = 0.0
    for lap in range(1, total_laps + 1):
        tire_age += 1
        temp_delta = track_temp - 30.0
        total_time += (
            base_lap
            + params.tire_base[tire]
            + params.temp_coeff[tire] * temp_delta * tire_age
            + degradation_for_age(tire_age, tire, params)
        )
        if lap in pit_map:
            total_time += pit_lane_time
            tire = pit_map[lap]
            tire_age = 0
    return total_time


def predict_order(race: dict, params: ModelParams) -> List[str]:
    totals: List[Tuple[str, float]] = []
    for strategy in race["strategies"].values():
        driver_id = strategy["driver_id"]
        totals.append((driver_id, simulate_driver_total_time(race["race_config"], strategy, params)))
    totals.sort(key=lambda x: x[1])
    return [d for d, _ in totals]


def kendall_like_score(pred: List[str], truth: List[str]) -> float:
    pos = {d: i for i, d in enumerate(pred)}
    n = len(truth)
    concordant = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if pos[truth[i]] < pos[truth[j]]:
                concordant += 1
    return concordant / total if total else 0.0


def evaluate(races: List[dict], params: ModelParams) -> Tuple[float, float]:
    exact = 0
    pairwise_sum = 0.0
    for race in races:
        pred = predict_order(race, params)
        truth = race["finishing_positions"]
        if pred == truth:
            exact += 1
        pairwise_sum += kendall_like_score(pred, truth)
    n = len(races)
    return exact / n, pairwise_sum / n


def sample_params(rng: random.Random) -> ModelParams:
    soft_base = rng.uniform(-1.6, -0.2)
    hard_base = rng.uniform(0.2, 1.8)

    soft_lin = rng.uniform(0.008, 0.03)
    med_lin = rng.uniform(0.006, soft_lin)
    hard_lin = rng.uniform(0.004, med_lin)

    soft_quad = rng.uniform(0.0001, 0.0012)
    med_quad = rng.uniform(0.00005, soft_quad)
    hard_quad = rng.uniform(0.00002, med_quad)

    soft_temp = rng.uniform(0.0005, 0.0035)
    med_temp = rng.uniform(0.0002, soft_temp)
    hard_temp = rng.uniform(0.0001, med_temp)

    return ModelParams(
        tire_base={"SOFT": soft_base, "MEDIUM": 0.0, "HARD": hard_base},
        temp_coeff={"SOFT": soft_temp, "MEDIUM": med_temp, "HARD": hard_temp},
        deg_lin={"SOFT": soft_lin, "MEDIUM": med_lin, "HARD": hard_lin},
        deg_quad={"SOFT": soft_quad, "MEDIUM": med_quad, "HARD": hard_quad},
        knee={
            "SOFT": rng.randint(6, 13),
            "MEDIUM": rng.randint(10, 18),
            "HARD": rng.randint(14, 24),
        },
    )


def mutate_params(base: ModelParams, rng: random.Random, scale: float = 0.2) -> ModelParams:
    def jitter(v: float, lo: float, hi: float) -> float:
        span = (hi - lo) * scale
        return max(lo, min(hi, v + rng.uniform(-span, span)))

    soft_base = jitter(base.tire_base["SOFT"], -1.6, -0.2)
    hard_base = jitter(base.tire_base["HARD"], 0.2, 1.8)

    soft_lin = jitter(base.deg_lin["SOFT"], 0.008, 0.03)
    med_lin = jitter(base.deg_lin["MEDIUM"], 0.006, soft_lin)
    hard_lin = jitter(base.deg_lin["HARD"], 0.004, med_lin)

    soft_quad = jitter(base.deg_quad["SOFT"], 0.0001, 0.0012)
    med_quad = jitter(base.deg_quad["MEDIUM"], 0.00005, soft_quad)
    hard_quad = jitter(base.deg_quad["HARD"], 0.00002, med_quad)

    soft_temp = jitter(base.temp_coeff["SOFT"], 0.0005, 0.0035)
    med_temp = jitter(base.temp_coeff["MEDIUM"], 0.0002, soft_temp)
    hard_temp = jitter(base.temp_coeff["HARD"], 0.0001, med_temp)

    return ModelParams(
        tire_base={"SOFT": soft_base, "MEDIUM": 0.0, "HARD": hard_base},
        temp_coeff={"SOFT": soft_temp, "MEDIUM": med_temp, "HARD": hard_temp},
        deg_lin={"SOFT": soft_lin, "MEDIUM": med_lin, "HARD": hard_lin},
        deg_quad={"SOFT": soft_quad, "MEDIUM": med_quad, "HARD": hard_quad},
        knee={
            "SOFT": max(6, min(13, base.knee["SOFT"] + rng.choice([-1, 0, 1]))),
            "MEDIUM": max(10, min(18, base.knee["MEDIUM"] + rng.choice([-1, 0, 1]))),
            "HARD": max(14, min(24, base.knee["HARD"] + rng.choice([-1, 0, 1]))),
        },
    )


def load_historical_races(data_dir: str, max_races: int) -> List[dict]:
    files = sorted(glob.glob(os.path.join(data_dir, "races_*.json")))
    races: List[dict] = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            batch = json.load(f)
            races.extend(batch)
        if len(races) >= max_races:
            break
    return races[:max_races]


def fmt(p: ModelParams) -> str:
    return json.dumps(
        {
            "tire_base": p.tire_base,
            "temp_coeff": p.temp_coeff,
            "deg_lin": p.deg_lin,
            "deg_quad": p.deg_quad,
            "knee": p.knee,
        },
        sort_keys=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit race model parameters")
    parser.add_argument("--data-dir", default="data/historical_races")
    parser.add_argument("--races", type=int, default=4000)
    parser.add_argument("--iters", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    races = load_historical_races(args.data_dir, args.races)

    split = int(len(races) * 0.8)
    train = races[:split]
    val = races[split:]

    best = sample_params(rng)
    best_train_exact, best_train_pair = evaluate(train, best)

    for i in range(args.iters):
        if i < args.iters // 2:
            cand = sample_params(rng)
        else:
            cand = mutate_params(best, rng, scale=max(0.05, 0.25 * (1.0 - i / args.iters)))

        exact, pair = evaluate(train, cand)
        score = exact * 4.0 + pair
        best_score = best_train_exact * 4.0 + best_train_pair
        if score > best_score:
            best = cand
            best_train_exact, best_train_pair = exact, pair
            val_exact, val_pair = evaluate(val, best)
            print(
                f"iter={i:04d} train_exact={best_train_exact:.4f} train_pair={best_train_pair:.4f} "
                f"val_exact={val_exact:.4f} val_pair={val_pair:.4f}"
            )

    val_exact, val_pair = evaluate(val, best)
    print("BEST_PARAMS")
    print(fmt(best))
    print(
        f"FINAL train_exact={best_train_exact:.4f} train_pair={best_train_pair:.4f} "
        f"val_exact={val_exact:.4f} val_pair={val_pair:.4f}"
    )


if __name__ == "__main__":
    main()