"""Byte-level regression harness for the solver.

Captures a canonical fingerprint of solve / solve_many / enumerate_moves over a
fixed battery of positions, plus full deterministic greedy-vs-greedy games.
Any solver-internal change (memo cache, numba/Cython DP) MUST reproduce this
fingerprint exactly — a different-but-equally-optimal arrangement would change
selected_indices → candidate ordering → the greedy opponent's move → the label
convention, invalidating every baseline. The paranoid crosscheck only compares
the objective value, so this harness is the real safety net.

Usage:
  python solver_regression.py capture golden.json   # BEFORE a change
  python solver_regression.py check   golden.json   # AFTER  a change
"""
import json
import random
import sys

from rummikub_solver import RummikubILPSolver, Tile, COLORS, flatten
from ppo_env import RummikubPPOEnv


def _canon_sets(sets):
    """Canonical, order-independent representation of an arrangement."""
    return sorted(sorted((t.color, t.number) for t in s) for s in sets)


def _canon_tiles(tiles):
    return sorted((t.color, t.number) for t in tiles)


def _fp_result(r):
    return {
        "status": r.status,
        "used": r.used_hand_tile_count,
        "sel_idx": list(r.selected_indices),
        "remaining": _canon_tiles(r.remaining_hand),
        "sets": _canon_sets(
            [s.completed_tiles for s in r.selected_sets]),
    }


def _random_position(rng):
    """A random (hand, table) — table is built from valid sets so it is legal."""
    full = [Tile(c, n) for c in COLORS for n in range(1, 14) for _ in range(2)]
    rng.shuffle(full)
    hand = full[: rng.randint(8, 16)]
    # build a small legal table from a few random valid sets out of the rest
    rest = full[16:]
    table = []
    # groups: same number, distinct colors. Deterministic: iterate numbers and
    # colors in fixed order (no set/hash-order dependence).
    by_num = {}
    for t in rest:
        by_num.setdefault(t.number, {})[t.color] = t
    for num in range(1, 14):
        bycol = by_num.get(num, {})
        cols = [c for c in COLORS if c in bycol]  # fixed COLORS order
        if len(cols) >= 3 and rng.random() < 0.3:
            k = rng.randint(3, len(cols))
            table.append([bycol[c] for c in cols[:k]])
    return hand, table[:4]


def battery(n=250, seed=12345):
    rng = random.Random(seed)
    solver = RummikubILPSolver(dp_crosscheck_every=0)
    out = []
    for _ in range(n):
        hand, table = _random_position(rng)
        for min_val, ignore in ((0, False), (30, True)):
            entry = {"min_val": min_val, "ignore": ignore,
                     "hand": _canon_tiles(hand), "table": _canon_sets(table)}
            r = solver.solve(hand_tiles=hand, table_sets=table,
                             require_use_at_least_one_hand_tile=False,
                             min_play_value=min_val, ignore_table=ignore)
            entry["solve"] = _fp_result(r)
            many = solver.solve_many(
                hand_tiles=hand, table_sets=table, max_solutions=20,
                require_use_at_least_one_hand_tile=True,
                min_play_value=min_val, ignore_table=ignore)
            entry["solve_many"] = [_fp_result(x) for x in many]
            enum = solver.enumerate_moves(
                hand_tiles=hand, table_sets=table,
                min_play_value=min_val, ignore_table=ignore, subset_limit=512)
            entry["enumerate"] = ([_fp_result(x) for x in enum]
                                  if enum is not None else None)
            out.append(entry)
    return out


def greedy_games(seeds=range(3000, 3020), meld=30):
    """Full deterministic greedy-vs-greedy games — end-to-end fingerprint."""
    games = []
    for sd in seeds:
        env = RummikubPPOEnv(max_candidates=20, max_turns=100, seed=sd,
                             ppo_player=0, opponent="ilp", initial_meld_value=meld)
        env.reset(seed=sd)
        moves, done = [], False
        # drive PPO seat with greedy (candidate 0 = max-play) too
        while not done:
            n = len(env.last_candidates)
            a = 0 if n > 0 else env.max_candidates
            _, _, term, trunc, info = env.step(a)
            moves.append((a, n, len(env.hands[0]), len(env.hands[1])))
            done = term or trunc
        games.append({"seed": sd, "outcome": info.get("outcome"),
                      "win_margin": info.get("win_margin", 0),
                      "loss_margin": info.get("loss_margin", 0),
                      "moves": moves})
    return games


def fingerprint():
    return {"battery": battery(), "games": greedy_games()}


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "capture":
        fp = fingerprint()
        with open(path, "w") as f:
            json.dump(fp, f)
        print(f"captured: {len(fp['battery'])} battery entries, "
              f"{len(fp['games'])} games -> {path}")
    elif mode == "check":
        with open(path) as f:
            golden = json.load(f)
        now = fingerprint()
        gb = json.dumps(golden["battery"], sort_keys=True)
        nb = json.dumps(now["battery"], sort_keys=True)
        gg = json.dumps(golden["games"], sort_keys=True)
        ng = json.dumps(now["games"], sort_keys=True)
        ok_b, ok_g = gb == nb, gg == ng
        print(f"battery: {'MATCH' if ok_b else 'MISMATCH'} "
              f"({len(now['battery'])} entries)")
        print(f"games  : {'MATCH' if ok_g else 'MISMATCH'} "
              f"({len(now['games'])} games)")
        if not (ok_b and ok_g):
            # find first differing battery entry for debugging
            for i, (a, b) in enumerate(zip(golden["battery"], now["battery"])):
                if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
                    print(f"first battery mismatch at #{i}: "
                          f"min_val={b['min_val']} ignore={b['ignore']}")
                    break
            sys.exit(1)
        print("ALL MATCH — solver behaviour unchanged.")
