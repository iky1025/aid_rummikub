"""R9: polynomial-time DP for single-turn Rummikub optimization.

Replaces the ILP on the hot path (van Rijn & Takes 2016 idea, adapted to
4 colors x 13 numbers x 2 copies, no jokers).

Sweep numbers 1..13. State = per color the multiset of open-run lengths,
capped at 3 ("safe"): a pair (l1<=l2), l in {0,1,2,3}. An open run must be
extended at each number or closed (closing requires length >= 3). Copies of
tile (c, n) not used for runs may form groups at n; group feasibility only
depends on the per-color usage vector g (precomputed over all 81 vectors).

Objective: maximize hand tiles used, subject to using ALL table tiles.
Optional: minimum value of played hand tiles (initial meld).
Traceback reconstructs a concrete arrangement (list of valid sets).
"""

from collections import Counter
from itertools import combinations

from rummikub_solver import COLORS, Tile


def _build_group_partitions():
    """gvec (per-color usage 0..2) -> concrete partition as list of
    color-index subsets (each a valid group of 3-4 distinct colors)."""
    subsets = [
        frozenset(c)
        for r in (3, 4)
        for c in combinations(range(4), r)
    ]
    options = [[]]
    options += [[s] for s in subsets]
    options += [[s1, s2] for s1 in subsets for s2 in subsets]
    parts = {}
    for opt in options:
        g = [0, 0, 0, 0]
        for s in opt:
            for c in s:
                g[c] += 1
        key = tuple(g)
        if max(g) <= 2 and key not in parts:
            parts[key] = opt
    return parts


GROUP_PARTS = _build_group_partitions()


def _build_run_transitions():
    """(pair, x) -> list of (new_pair, extended_capped_lengths, n_new).

    pair: sorted (l1, l2) open-run lengths (0 = no run), capped at 3.
    x: copies of the current tile placed into runs.
    Every open run must be extended or closed; closing requires length 3.
    """
    trans = {}
    pairs = [(a, b) for a in range(4) for b in range(a, 4)]
    for pair in pairs:
        runs = [l for l in pair if l > 0]
        for x in range(3):
            outs = {}
            for k in range(min(x, len(runs)) + 1):
                for ext in combinations(range(len(runs)), k):
                    closed = [runs[i] for i in range(len(runs)) if i not in ext]
                    if any(l < 3 for l in closed):
                        continue
                    n_new = x - k
                    new_open = [min(runs[i] + 1, 3) for i in ext] + [1] * n_new
                    if len(new_open) > 2:
                        continue
                    new_pair = tuple(sorted(new_open + [0] * (2 - len(new_open))))
                    key = (new_pair,
                           tuple(sorted(runs[i] for i in ext)), n_new)
                    outs[key] = True
            trans[(pair, x)] = list(outs)
    return trans


RUN_TRANS = _build_run_transitions()
EMPTY_PAIR = (0, 0)
SAFE = {0, 3}
_MISS = object()  # sentinel: distinguishes "not cached" from a cached None


class RummikubDP:
    """Single-turn optimizer. Counts are per (color_idx, number)."""

    def __init__(self):
        # B-0 (2026-07-19): memoize solve. Profiling showed the DP dominates
        # teacher-decision time and many calls repeat (the determinizations and
        # enumerate_moves re-solve identical (hand, table) positions; feasible()
        # routes through solve() too). Keyed on the exact multiset inputs. The
        # cached (used, sets) is never mutated by any caller — every consumer
        # copies via list(s) / Counter(s) / flatten (audited) — so sharing the
        # object is safe. Verified byte-identical via solver_regression.py.
        self._cache = {}
        self._cache_cap = 1_000_000

    def solve(self, hand_counter, table_counter, min_play_value=0):
        """Maximize hand tiles used; all table tiles must be used.

        Returns (used_hand_count, sets) where sets is a list of tile lists,
        or None if no arrangement exists (even using zero hand tiles this
        can fail only if the table itself is corrupt).
        min_play_value > 0: only count solutions whose played-hand value
        meets the threshold (returns best such, else None).
        """
        ckey = (frozenset(hand_counter.items()),
                frozenset(table_counter.items()), min_play_value)
        cached = self._cache.get(ckey, _MISS)
        if cached is not _MISS:
            return cached
        result = self._solve_uncached(hand_counter, table_counter, min_play_value)
        if len(self._cache) >= self._cache_cap:
            self._cache.clear()
        self._cache[ckey] = result
        return result

    def _solve_uncached(self, hand_counter, table_counter, min_play_value=0):
        avail = {}
        mand = {}
        for c in range(4):
            for n in range(1, 14):
                tile = Tile(COLORS[c], n)
                t = table_counter.get(tile, 0)
                h = hand_counter.get(tile, 0)
                mand[(c, n)] = t
                avail[(c, n)] = t + h

        cap_v = min_play_value if min_play_value > 0 else 0

        # layer: state -> {(tiles, value_capped): parent_record}
        # parent_record: (prev_state, prev_key, decisions, gvec)
        start_state = (EMPTY_PAIR,) * 4
        layer = {start_state: {(0, 0): None}}
        parents = []  # parents[n-1][(state, key)] = record

        for n in range(1, 14):
            new_layer = {}
            layer_parents = {}
            for state, keymap in layer.items():
                per_color = []
                dead = False
                for c in range(4):
                    a = avail[(c, n)]
                    tb = mand[(c, n)]
                    opts = []
                    for u in range(tb, a + 1):
                        for x in range(0, u + 1):
                            g = u - x
                            if g > 2:
                                continue
                            for (npair, extl, n_new) in RUN_TRANS[(state[c], x)]:
                                opts.append(
                                    (npair, g, u - tb, (x, extl, n_new))
                                )
                    if not opts:
                        dead = True
                        break
                    per_color.append(opts)
                if dead:
                    continue

                for o0 in per_color[0]:
                    for o1 in per_color[1]:
                        for o2 in per_color[2]:
                            g012 = o0[1] + o1[1] + o2[1]
                            for o3 in per_color[3]:
                                gvec = (o0[1], o1[1], o2[1], o3[1])
                                if g012 + o3[1] > 0 and gvec not in GROUP_PARTS:
                                    continue
                                nstate = (o0[0], o1[0], o2[0], o3[0])
                                gain = o0[2] + o1[2] + o2[2] + o3[2]
                                vgain = n * gain
                                decisions = (o0[3], o1[3], o2[3], o3[3])
                                tgt = new_layer.setdefault(nstate, {})
                                tp = layer_parents
                                for (tiles, val) in keymap:
                                    nk = (
                                        tiles + gain,
                                        min(cap_v, val + vgain) if cap_v else 0,
                                    )
                                    if nk not in tgt:
                                        tgt[nk] = True
                                        tp[(nstate, nk)] = (
                                            state, (tiles, val),
                                            decisions, gvec,
                                        )
            parents.append(layer_parents)
            layer = new_layer
            if not layer:
                return None

        # final: all runs safe; pick max tiles meeting value threshold
        best = None
        for state, keymap in layer.items():
            if any(l not in SAFE for pair in state for l in pair):
                continue
            for (tiles, val) in keymap:
                if cap_v and val < cap_v:
                    continue
                if best is None or tiles > best[2]:
                    best = (state, (tiles, val), tiles)
        if best is None:
            return None

        sets = self._traceback(best[0], best[1], parents)
        return best[2], sets

    def feasible(self, mandatory_counter):
        """Can this exact multiset be fully partitioned into valid sets?
        Returns a concrete arrangement (list of tile lists) or None."""
        res = self.solve(Counter(), mandatory_counter)
        return res[1] if res is not None else None

    def _traceback(self, state, key, parents):
        # walk back collecting decisions per number
        chain = [None] * 13  # (decisions, gvec) at each n
        s, k = state, key
        for n in range(13, 0, -1):
            rec = parents[n - 1][(s, k)]
            prev_state, prev_key, decisions, gvec = rec
            chain[n - 1] = (decisions, gvec)
            s, k = prev_state, prev_key

        # replay forward, tracking actual runs as [start, length]
        sets = []
        open_runs = [[] for _ in range(4)]
        for n in range(1, 14):
            decisions, gvec = chain[n - 1]
            for c in range(4):
                x, extl, n_new = decisions[c]
                remaining = list(open_runs[c])
                kept = []
                for capped in extl:
                    for i, run in enumerate(remaining):
                        if min(run[1], 3) == capped:
                            run[1] += 1
                            kept.append(run)
                            del remaining[i]
                            break
                # runs not extended close before n
                for start, length in remaining:
                    sets.append(
                        [Tile(COLORS[c], m) for m in range(start, start + length)]
                    )
                for _ in range(n_new):
                    kept.append([n, 1])
                open_runs[c] = kept
            if any(gvec):
                for subset in GROUP_PARTS[gvec]:
                    sets.append([Tile(COLORS[c], n) for c in sorted(subset)])

        for c in range(4):
            for start, length in open_runs[c]:
                sets.append(
                    [Tile(COLORS[c], m) for m in range(start, start + length)]
                )
        return sets
