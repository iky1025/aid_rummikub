import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
from datetime import datetime
from pathlib import Path

import torch

from ppo_env import RummikubPPOEnv
from ppo_model import ActorCritic


_WORKER_MODEL = None
_WORKER_DEVICE = torch.device("cpu")


def _init_eval_worker(state_dict, max_candidates):
    global _WORKER_MODEL

    torch.set_num_threads(1)
    if state_dict is None:
        _WORKER_MODEL = None
        return

    _WORKER_MODEL = ActorCritic(
        obs_dim=RummikubPPOEnv.OBS_DIM,
        cand_feat_dim=RummikubPPOEnv.CAND_FEAT_DIM,
        max_candidates=max_candidates,
    )
    _WORKER_MODEL.load_state_dict(state_dict)
    _WORKER_MODEL.eval()


def _run_game_worker(task):
    policy, deal_index, game_seed, ppo_player, max_candidates, max_turns = task
    result = _run_game(
        model=_WORKER_MODEL,
        device=_WORKER_DEVICE,
        game_seed=game_seed,
        ppo_player=ppo_player,
        max_candidates=max_candidates,
        max_turns=max_turns,
    )
    return {
        "policy": policy,
        "deal_index": deal_index,
        "game_seed": game_seed,
        "ppo_player": ppo_player,
        "result": result,
    }


def _wilson_interval(successes, total, z=1.96):
    if total == 0:
        return 0.0, 0.0

    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - margin, center + margin


def _choose_action(model, obs, cand_feats, mask, device):
    if model is None:
        return 0 if mask[0] == 1 else len(mask) - 1

    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    cand_tensor = torch.tensor(cand_feats, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model.forward_actor(
            obs_tensor.unsqueeze(0), cand_tensor.unsqueeze(0)
        ).squeeze(0)
        logits[mask_tensor == 0] = -1e9
        return torch.argmax(logits).item()


def _run_game(model, device, game_seed, ppo_player, max_candidates, max_turns):
    env = RummikubPPOEnv(
        max_candidates=max_candidates,
        max_turns=max_turns,
        seed=game_seed,
        ppo_player=ppo_player,
    )
    env.reset()

    done = env.is_done()
    episode_reward = 0.0
    steps = 0
    infos = []
    final_info = env.get_info()
    if done:
        if final_info["winner"] == "ppo":
            episode_reward = env.win_reward
        elif final_info["winner"] == "ilp":
            episode_reward = -env.win_reward

    while not done:
        obs, cand_feats, mask = env.get_policy_inputs()
        action = _choose_action(model, obs, cand_feats, mask, device)
        _, reward, done, info = env.step(action)
        episode_reward += reward
        steps += 1
        infos.append(info)
        final_info = info

    return {
        "winner": final_info["winner"],
        "timeout": final_info["timeout"],
        "reward": episode_reward,
        "steps": steps,
        "ppo_hand_count": final_info["ppo_hand_count"],
        "ilp_hand_count": final_info["ilp_hand_count"],
        "ppo_initial_meld_done": final_info["ppo_initial_meld_done"],
        "ilp_initial_meld_done": final_info["ilp_initial_meld_done"],
        "candidate_count": sum(x["candidate_count"] for x in infos),
        "raw_candidate_count": sum(x["raw_candidate_count"] for x in infos),
        "pool_candidate_count": sum(x["pool_candidate_count"] for x in infos),
        "duplicate_candidate_count": sum(
            x["duplicate_candidate_count"] for x in infos
        ),
        "strategy_candidate_count": sum(
            x["strategy_candidate_count"] for x in infos
        ),
        "ilp_used_hand_tiles": sum(x["ilp_used_hand_tiles"] for x in infos),
        "turn_info_count": len(infos),
    }


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_run_id(model_path, deals, seed, max_turns):
    config = {
        "model_sha256": _file_sha256(model_path),
        "deals": deals,
        "seed": seed,
        "max_turns": max_turns,
        "obs_dim": RummikubPPOEnv.OBS_DIM,
        "cand_feat_dim": RummikubPPOEnv.CAND_FEAT_DIM,
        "reward_version": RummikubPPOEnv.REWARD_VERSION,
    }
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _record_key(record):
    return (
        record["policy"],
        int(record["deal_index"]),
        int(record["ppo_player"]),
    )


def _load_records(results_path, run_id):
    records = {}
    path = Path(results_path)
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"ignoring incomplete result line={line_number}", flush=True)
                continue
            if record.get("run_id") == run_id:
                records[_record_key(record)] = record
    return records


def _append_record(results_path, record):
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record, ensure_ascii=True) + "\n").encode("utf-8")
    with path.open("ab+") as file:
        file.seek(0, os.SEEK_END)
        if file.tell() > 0:
            file.seek(-1, os.SEEK_END)
            if file.read(1) != b"\n":
                file.write(b"\n")
        file.write(encoded)
        file.flush()
        os.fsync(file.fileno())


def _write_json(path, data):
    if not path:
        return True
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    last_error = None
    for attempt in range(10):
        try:
            with temporary.open("w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=True, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, target)
            return True
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.1 * (attempt + 1))

    print(
        f"warning: could not update {target} after retries: {last_error}",
        flush=True,
    )
    return False


def _format_duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _save_progress(tracker, policy, policy_completed, policy_total):
    elapsed = time.monotonic() - tracker["started_at"]
    new_completed = tracker["completed"] - tracker["initial_completed"]
    rate = new_completed / elapsed if elapsed > 0 and new_completed > 0 else 0.0
    remaining = tracker["total"] - tracker["completed"]
    eta_seconds = remaining / rate if rate > 0 else None
    now = datetime.now().astimezone().isoformat()
    progress = {
        "run_id": tracker["run_id"],
        "stage": "evaluating",
        "policy": policy,
        "policy_completed": policy_completed,
        "policy_total": policy_total,
        "completed_games": tracker["completed"],
        "total_games": tracker["total"],
        "new_games_this_run": new_completed,
        "games_per_minute": rate * 60.0,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta_seconds,
        "updated_at": now,
    }
    _write_json(tracker["progress_path"], progress)
    _write_json(
        tracker["status_path"],
        {
            "stage": "evaluating",
            "detail": (
                f"{tracker['completed']}/{tracker['total']} games; "
                f"policy={policy}; eta={_format_duration(eta_seconds)}"
            ),
            "run_id": tracker["run_id"],
            "updated_at": now,
        },
    )
    return rate, eta_seconds


def _print_summary(label, deals, results):
    games = len(results)
    wins = sum(x["winner"] == "ppo" for x in results)
    losses = sum(x["winner"] == "ilp" for x in results)
    timeouts = sum(x["timeout"] for x in results)
    decisive_games = wins + losses
    ci_low, ci_high = _wilson_interval(wins, games)
    turn_infos = sum(x["turn_info_count"] for x in results)

    def average(field, denominator=games):
        return sum(x[field] for x in results) / denominator if denominator else 0.0

    print(f"\n[{label}]")
    print(f"deals={deals}")
    print(f"paired_games={games}")
    print(f"ppo_win_count={wins}")
    print(f"ilp_win_count={losses}")
    print(f"draw_or_timeout_count={timeouts}")
    print(f"win_rate={wins / games:.3f}")
    print(f"win_rate_95ci=[{ci_low:.3f}, {ci_high:.3f}]")
    decisive_rate = wins / decisive_games if decisive_games else 0.0
    print(f"decisive_win_rate={decisive_rate:.3f}")
    print(f"avg_reward={average('reward'):.3f}")
    print(f"avg_steps={average('steps'):.2f}")
    print(f"avg_final_ppo_hand_count={average('ppo_hand_count'):.2f}")
    print(f"avg_final_ilp_hand_count={average('ilp_hand_count'):.2f}")
    print(
        "avg_remaining_hand_gap="
        f"{average('ppo_hand_count') - average('ilp_hand_count'):.2f}"
    )
    if turn_infos:
        print(
            "avg_unique_candidate_count="
            f"{sum(x['candidate_count'] for x in results) / turn_infos:.2f}"
        )
        print(
            "avg_raw_candidate_count="
            f"{sum(x['raw_candidate_count'] for x in results) / turn_infos:.2f}"
        )
        print(
            "avg_candidate_pool_count="
            f"{sum(x['pool_candidate_count'] for x in results) / turn_infos:.2f}"
        )
        print(
            "avg_duplicate_candidate_count="
            f"{sum(x['duplicate_candidate_count'] for x in results) / turn_infos:.2f}"
        )
        print(
            "avg_strategy_solution_count="
            f"{sum(x['strategy_candidate_count'] for x in results) / turn_infos:.2f}"
        )
        print(
            "avg_ilp_used_hand_tiles="
            f"{sum(x['ilp_used_hand_tiles'] for x in results) / turn_infos:.2f}"
        )
    print(
        "ppo_initial_meld_rate="
        f"{sum(x['ppo_initial_meld_done'] for x in results) / games:.3f}"
    )
    print(
        "ilp_initial_meld_rate="
        f"{sum(x['ilp_initial_meld_done'] for x in results) / games:.3f}"
    )


def _evaluate_policy(
    policy,
    label,
    model,
    device,
    deals,
    seed,
    max_candidates,
    max_turns,
    workers,
    worker_state_dict,
    run_id,
    results_path,
    records,
    tracker,
    progress_every,
):
    all_tasks = [
        (policy, deal_index, seed + deal_index, ppo_player, max_candidates, max_turns)
        for deal_index in range(deals)
        for ppo_player in (0, 1)
    ]
    saved = {
        key: record
        for key, record in records.items()
        if key[0] == policy
    }
    pending = [
        task for task in all_tasks if (policy, task[1], task[3]) not in saved
    ]
    policy_total = len(all_tasks)
    print(
        f"[{policy}] resumed={len(saved)} pending={len(pending)} "
        f"total={policy_total}",
        flush=True,
    )

    def completed_records():
        if workers > 1:
            context = mp.get_context("spawn")
            with context.Pool(
                processes=workers,
                initializer=_init_eval_worker,
                initargs=(worker_state_dict, max_candidates),
            ) as pool:
                yield from pool.imap_unordered(
                    _run_game_worker, pending, chunksize=1
                )
        else:
            for task in pending:
                policy_name, deal_index, game_seed, ppo_player, candidate_count, turns = task
                result = _run_game(
                    model=model,
                    device=device,
                    game_seed=game_seed,
                    ppo_player=ppo_player,
                    max_candidates=candidate_count,
                    max_turns=turns,
                )
                yield {
                    "policy": policy_name,
                    "deal_index": deal_index,
                    "game_seed": game_seed,
                    "ppo_player": ppo_player,
                    "result": result,
                }

    if not pending:
        _save_progress(tracker, policy, len(saved), policy_total)

    for completed in completed_records():
        record = {"run_id": run_id, **completed}
        key = _record_key(record)
        if key in records:
            continue
        _append_record(results_path, record)
        records[key] = record
        saved[key] = record
        tracker["completed"] += 1
        rate, eta_seconds = _save_progress(
            tracker, policy, len(saved), policy_total
        )
        if tracker["completed"] % progress_every == 0 or len(saved) == policy_total:
            print(
                f"progress={tracker['completed']}/{tracker['total']} "
                f"rate={rate * 60.0:.2f} games/min "
                f"eta={_format_duration(eta_seconds)}",
                flush=True,
            )

    ordered = [
        saved[(policy, deal_index, ppo_player)]["result"]
        for deal_index in range(deals)
        for ppo_player in (0, 1)
    ]
    _print_summary(label, deals, ordered)
    return ordered


def evaluate(
    model_path="rummikub_ppo_model.pt",
    episodes=50,
    seed=123,
    include_greedy_baseline=True,
    workers=None,
    max_turns=100,
    results_path="eval_results.jsonl",
    progress_path="eval_progress.json",
    status_path="experiment_status.json",
    progress_every=50,
):
    if workers is None:
        workers = max(1, min(4, (os.cpu_count() or 2) // 2))
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if progress_every < 1:
        raise ValueError("progress_every must be at least 1")

    model_path = str(Path(model_path).resolve())
    run_id = _make_run_id(model_path, episodes, seed, max_turns)
    records = _load_records(results_path, run_id)
    policies = ["ppo"] + (["greedy"] if include_greedy_baseline else [])
    valid_keys = {
        (policy, deal_index, ppo_player)
        for policy in policies
        for deal_index in range(episodes)
        for ppo_player in (0, 1)
    }
    records = {key: value for key, value in records.items() if key in valid_keys}
    total_games = len(valid_keys)
    tracker = {
        "run_id": run_id,
        "total": total_games,
        "completed": len(records),
        "initial_completed": len(records),
        "started_at": time.monotonic(),
        "progress_path": progress_path,
        "status_path": status_path,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_candidates = 10
    state_dict = torch.load(model_path, map_location="cpu")
    model = ActorCritic(
        obs_dim=RummikubPPOEnv.OBS_DIM,
        cand_feat_dim=RummikubPPOEnv.CAND_FEAT_DIM,
        max_candidates=max_candidates,
    ).to(device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "checkpoint is incompatible with the current observation or "
            "candidate features; retrain the model with the updated environment"
        ) from exc
    model.eval()
    print(f"evaluation_run_id={run_id}")
    print(f"evaluation_workers={workers}")
    print(f"saved_games={len(records)} total_games={total_games}")
    if workers > 1 and device.type == "cuda":
        print("parallel evaluation uses CPU models in worker processes")

    _evaluate_policy(
        policy="ppo",
        label="PPO+ILP candidates vs ILP",
        model=model,
        device=device,
        deals=episodes,
        seed=seed,
        max_candidates=max_candidates,
        max_turns=max_turns,
        workers=workers,
        worker_state_dict=state_dict,
        run_id=run_id,
        results_path=results_path,
        records=records,
        tracker=tracker,
        progress_every=progress_every,
    )

    if include_greedy_baseline:
        _evaluate_policy(
            policy="greedy",
            label="greedy top-1 vs ILP baseline",
            model=None,
            device=device,
            deals=episodes,
            seed=seed,
            max_candidates=1,
            max_turns=max_turns,
            workers=workers,
            worker_state_dict=None,
            run_id=run_id,
            results_path=results_path,
            records=records,
            tracker=tracker,
            progress_every=progress_every,
        )

    now = datetime.now().astimezone().isoformat()
    _write_json(
        progress_path,
        {
            "run_id": run_id,
            "stage": "complete",
            "completed_games": total_games,
            "total_games": total_games,
            "updated_at": now,
        },
    )
    _write_json(
        status_path,
        {
            "stage": "complete",
            "detail": f"evaluation completed: {total_games}/{total_games} games",
            "run_id": run_id,
            "updated_at": now,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="rummikub_ppo_model.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--results-path", default="eval_results.jsonl")
    parser.add_argument("--progress-path", default="eval_progress.json")
    parser.add_argument("--status-path", default="experiment_status.json")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--no-greedy-baseline", action="store_true")
    args = parser.parse_args()
    evaluate(
        model_path=args.model_path,
        episodes=args.episodes,
        seed=args.seed,
        include_greedy_baseline=not args.no_greedy_baseline,
        workers=args.workers,
        max_turns=args.max_turns,
        results_path=args.results_path,
        progress_path=args.progress_path,
        status_path=args.status_path,
        progress_every=args.progress_every,
    )
