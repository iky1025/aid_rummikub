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
from itertools import combinations, combinations_with_replacement, product

from rummikub_solver import COLORS, JOKER, Tile


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


def _build_group_parts_j():
    """R11 jokers. (g0,g1,g2,g3,jg) -> a partition of the number-n group tiles
    into valid groups, using exactly the real per-color counts g (each 0..2) and
    exactly jg jokers (0..2), as a list of (frozenset real color indices,
    n_jokers), or None if impossible.

    A group is 3-4 tiles of distinct real colors + jokers (a joker stands for a
    missing color), size 3-4. The jg=0 slice is identical to GROUP_PARTS
    (verified: 0 mismatches, scratchpad/group_joker.py brute-force cross-check)."""
    shapes = []                                  # (frozenset real colors, jokers)
    for r in range(0, 5):
        for cols in combinations(range(4), r):
            for j in range(0, 4 - r + 1):
                if 3 <= r + j <= 4:
                    shapes.append((frozenset(cols), j))
    table = {}
    for g in product(range(3), repeat=4):
        for jg in range(3):
            found = None
            for k in range(0, 5):
                if found is not None:
                    break
                for combo in combinations_with_replacement(range(len(shapes)), k):
                    use = [0, 0, 0, 0]
                    jok = 0
                    for idx in combo:
                        cols, j = shapes[idx]
                        for c in cols:
                            use[c] += 1
                        jok += j
                    if tuple(use) == g and jok == jg:
                        found = [shapes[idx] for idx in combo]
                        break
            table[(*g, jg)] = found
    return table


GROUP_PARTS_J = _build_group_parts_j()


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
        # x up to 4 (2 real + 2 joker copies of one tile). x >= 3 always yields
        # an empty transition list (>2 open runs per color is pruned below), so
        # this only ADDS empty entries — the x in {0,1,2} keys stay identical, so
        # the verified jokerless path is unchanged — while avoiding a KeyError
        # when a joker pushes run-tiles past 2 (and fixing a latent jokerless
        # crash on 3+ copies of a tile).
        for x in range(5):
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
        # Cap bounds memory: cleared wholesale on overflow. Behaviour-neutral —
        # only affects when a cache miss recurs, never a return value. 200k keeps
        # the hot within-decision working set (where nearly all of the 54%
        # redundancy lives, used within seconds), so the 2.3x speedup is
        # preserved; the long cross-game tail is dropped. 1M let 4 workers grow
        # to ~8GB collectively and OOM before any single cache hit its cap.
        self._cache_cap = 200_000

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
        jt = table_counter.get(JOKER, 0)
        jh = hand_counter.get(JOKER, 0)
        if jt + jh == 0:
            return self._solve_jokerless(
                hand_counter, table_counter, min_play_value)
        return self._solve_joker(
            hand_counter, table_counter, min_play_value, jt, jh)

    def solve_all_k(self, hand_counter, table_counter, min_play_value=0):
        """Tile-count diversification from a SINGLE sweep: {k: sets} giving, for
        each achievable hand-tile count k >= 1, one valid full arrangement that
        uses exactly k hand tiles (all table tiles used, played value >=
        min_play_value). Replaces solve_many Phase 1's per-k ILP exact-k calls.

        Joker positions support min_play_value == 0 only (meld+joker stays on
        the ILP). Not memoized (distinct return shape from solve)."""
        jt = table_counter.get(JOKER, 0)
        jh = hand_counter.get(JOKER, 0)
        if jt + jh == 0:
            swept = self._sweep_jokerless(
                hand_counter, table_counter, min_play_value)
            if swept is None:
                return {}
            layer, parents, cap_v = swept
            return self._all_k_jokerless(layer, parents, cap_v)
        assert min_play_value == 0, "joker solve_all_k handles max-play only"
        swept = self._sweep_joker(hand_counter, table_counter, jt, jh)
        if swept is None:
            return {}
        layer, parents, J = swept
        return self._all_k_joker(layer, parents, jt)

    def generate_moves(self, hand_counter, table_counter, min_play_value=0):
        """R11 generating DP: enumerate ALL distinct playable moves in one
        memoized sweep — no ILP exclusions, no 2^n sub-multiset iteration.

        A move is a played hand sub-multiset S (Counter of Tiles; JOKER entries
        are hand jokers played) such that table + S partitions into valid sets
        (all table tiles used). Returns the list of such Counters, most tiles
        first. Memoizes, per (number, run-state[, jokers-placed]), the SET of
        distinct played-suffix multisets that complete validly; the distinct-
        move count is small so those sets stay small and are shared across
        arrangements. Complete (matches enumerate_moves with 0 mismatch) and
        fast even with jokers, where sub-multiset enumeration blows up.

        Jokered supports min_play_value == 0 only (joker meld value is not
        modelled, same limitation as the joker DP)."""
        jt = table_counter.get(JOKER, 0)
        jh = hand_counter.get(JOKER, 0)
        if jt + jh == 0:
            raw = self._gen_jokerless(hand_counter, table_counter, min_play_value)
            moves = [Counter({Tile(COLORS[c], n): p for (c, n), p in S})
                     for S in raw]
        else:
            assert min_play_value == 0, "joker generate_moves is max-play only"
            moves = []
            for rp, hand_jokers in self._gen_joker(
                    hand_counter, table_counter, jt, jh):
                m = Counter({Tile(COLORS[c], n): p for (c, n), p in rp})
                if hand_jokers:
                    m[JOKER] = hand_jokers
                moves.append(m)
        moves.sort(key=lambda m: -sum(m.values()))
        return moves

    def _gen_jokerless(self, hand_counter, table_counter, min_play_value):
        avail = {}
        mand = {}
        for c in range(4):
            for n in range(1, 14):
                tile = Tile(COLORS[c], n)
                t = table_counter.get(tile, 0)
                avail[(c, n)] = t + hand_counter.get(tile, 0)
                mand[(c, n)] = t
        memo = {}

        def suffixes(n, state):
            # set of frozenset(((c,n),p)) played-suffixes (numbers >= n) that
            # complete to an all-valid table.
            if n == 14:
                if all(l in SAFE for pair in state for l in pair):
                    return frozenset({frozenset()})
                return frozenset()
            key = (n, state)
            got = memo.get(key)
            if got is not None:
                return got
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
                            opts.append((npair, g, u - tb))
                if not opts:
                    dead = True
                    break
                per_color.append(opts)
            out = set()
            if not dead:
                for o0 in per_color[0]:
                    for o1 in per_color[1]:
                        for o2 in per_color[2]:
                            for o3 in per_color[3]:
                                gvec = (o0[1], o1[1], o2[1], o3[1])
                                if any(gvec) and gvec not in GROUP_PARTS:
                                    continue
                                sub = suffixes(
                                    n + 1, (o0[0], o1[0], o2[0], o3[0]))
                                if not sub:
                                    continue
                                here = [((c, n), o[2]) for c, o in
                                        enumerate((o0, o1, o2, o3)) if o[2]]
                                if not here:
                                    out |= sub
                                else:
                                    for s in sub:
                                        d = dict(s)
                                        for k, v in here:
                                            d[k] = d.get(k, 0) + v
                                        out.add(frozenset(d.items()))
            out = frozenset(out)
            memo[key] = out
            return out

        res = []
        for S in suffixes(1, (EMPTY_PAIR,) * 4):
            if not S:
                continue
            if min_play_value and sum(n * p for (c, n), p in S) < min_play_value:
                continue
            res.append(S)
        return res

    def _gen_joker(self, hand_counter, table_counter, jt, jh):
        J = jt + jh
        avail = {}
        mand = {}
        for c in range(4):
            for n in range(1, 14):
                tile = Tile(COLORS[c], n)
                t = table_counter.get(tile, 0)
                avail[(c, n)] = t + hand_counter.get(tile, 0)
                mand[(c, n)] = t
        memo = {}

        def suffixes(n, state, j):
            # set of (real_plays_frozenset, j_final): distinct played-suffixes
            # (real tiles) with the total jokers placed by the end.
            if n == 14:
                if j >= jt and all(l in SAFE for pair in state for l in pair):
                    return frozenset({(frozenset(), j)})
                return frozenset()
            key = (n, state, j)
            got = memo.get(key)
            if got is not None:
                return got
            budget = J - j
            per_color = []
            dead = False
            for c in range(4):
                a = avail[(c, n)]
                tb = mand[(c, n)]
                opts = []
                for u in range(tb, a + 1):
                    for x_real in range(0, u + 1):
                        g = u - x_real
                        if g > 2:
                            continue
                        for rj in range(0, budget + 1):
                            trans = RUN_TRANS.get((state[c], x_real + rj))
                            if not trans:
                                continue
                            for (npair, extl, n_new) in trans:
                                opts.append((npair, g, u - tb, rj))
                if not opts:
                    dead = True
                    break
                per_color.append(opts)
            out = set()
            if not dead:
                for o0 in per_color[0]:
                    for o1 in per_color[1]:
                        for o2 in per_color[2]:
                            for o3 in per_color[3]:
                                gvec = (o0[1], o1[1], o2[1], o3[1])
                                run_jok = o0[3] + o1[3] + o2[3] + o3[3]
                                if j + run_jok > J:
                                    continue
                                nstate = (o0[0], o1[0], o2[0], o3[0])
                                here = [((c, n), o[2]) for c, o in
                                        enumerate((o0, o1, o2, o3)) if o[2]]
                                for jg in range(0, J - j - run_jok + 1):
                                    if (any(gvec) or jg) and \
                                            GROUP_PARTS_J.get((*gvec, jg)) is None:
                                        continue
                                    sub = suffixes(
                                        n + 1, nstate, j + run_jok + jg)
                                    if not sub:
                                        continue
                                    for (rp, jf) in sub:
                                        if here:
                                            d = dict(rp)
                                            for k, v in here:
                                                d[k] = d.get(k, 0) + v
                                            out.add((frozenset(d.items()), jf))
                                        else:
                                            out.add((rp, jf))
            out = frozenset(out)
            memo[key] = out
            return out

        res = []
        for (rp, jf) in suffixes(1, (EMPTY_PAIR,) * 4, 0):
            hand_jokers = jf - jt
            if sum(p for _, p in rp) + hand_jokers <= 0:
                continue
            res.append((rp, hand_jokers))
        return res

    def _solve_jokerless(self, hand_counter, table_counter, min_play_value=0):
        swept = self._sweep_jokerless(
            hand_counter, table_counter, min_play_value)
        if swept is None:
            return None
        layer, parents, cap_v = swept
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
        return best[2], self._traceback(best[0], best[1], parents)

    def _all_k_jokerless(self, layer, parents, cap_v):
        """One valid arrangement per achievable hand-tile count k (>=1), read
        off a single completed sweep. k here == tiles (real hand tiles used)."""
        per_k = {}
        for state, keymap in layer.items():
            if any(l not in SAFE for pair in state for l in pair):
                continue
            for (tiles, val) in keymap:
                if cap_v and val < cap_v:
                    continue
                if tiles <= 0 or tiles in per_k:
                    continue
                per_k[tiles] = (state, (tiles, val))
        return {k: self._traceback(s, key, parents)
                for k, (s, key) in per_k.items()}

    def _sweep_jokerless(self, hand_counter, table_counter, min_play_value=0):
        """Shared number-sweep (jokerless). Returns (final_layer, parents,
        cap_v), or None if the table itself cannot be arranged. `solve` and
        `solve_all_k` both read the same completed layer."""
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
        return layer, parents, cap_v

    def _solve_joker(self, hand_counter, table_counter, min_play_value, jt, jh):
        """R11 max-play with jokers (min_play_value must be 0 — the meld case
        stays on the ILP, routed in rummikub_solver.solve). J = jt+jh jokers are
        wildcards: a run-joker acts as (color, n) for one color's run; a
        group-joker fills a missing color in a group (GROUP_PARTS_J). The state
        carries `j` = jokers placed so far; table jokers (jt) are mandatory, so a
        final state needs j >= jt, and the played-hand count is
        real_hand_tiles + (j - jt) hand jokers."""
        assert min_play_value == 0, "joker DP handles max-play only"
        swept = self._sweep_joker(hand_counter, table_counter, jt, jh)
        if swept is None:
            return None
        layer, parents, J = swept
        best = None
        for state, keymap in layer.items():
            if any(l not in SAFE for pair in state[:4] for l in pair):
                continue
            j = state[4]
            if j < jt:
                continue
            for tiles in keymap:
                total = tiles + (j - jt)
                if best is None or total > best[1]:
                    best = (state, total, tiles)
        if best is None:
            return None
        return best[1], self._traceback_joker(best[0], best[2], parents)

    def _all_k_joker(self, layer, parents, jt):
        """One valid arrangement per achievable hand-tile count k (>=1). Here
        k == real hand tiles + (jokers placed - table jokers)."""
        per_k = {}
        for state, keymap in layer.items():
            if any(l not in SAFE for pair in state[:4] for l in pair):
                continue
            j = state[4]
            if j < jt:
                continue
            for tiles in keymap:
                total = tiles + (j - jt)
                if total <= 0 or total in per_k:
                    continue
                per_k[total] = (state, tiles)
        return {k: self._traceback_joker(s, tk, parents)
                for k, (s, tk) in per_k.items()}

    def _sweep_joker(self, hand_counter, table_counter, jt, jh):
        """Shared jokered max-play sweep. Returns (final_layer, parents, J) or
        None. State carries j = jokers placed; table jokers (jt) are mandatory."""
        J = jt + jh
        avail = {}
        mand = {}
        for c in range(4):
            for n in range(1, 14):
                tile = Tile(COLORS[c], n)
                t = table_counter.get(tile, 0)
                h = hand_counter.get(tile, 0)
                mand[(c, n)] = t
                avail[(c, n)] = t + h

        start_state = (EMPTY_PAIR,) * 4 + (0,)
        layer = {start_state: {0: None}}         # state -> {tiles_real: parent}
        parents = []

        for n in range(1, 14):
            new_layer = {}
            layer_parents = {}
            for state, keymap in layer.items():
                j = state[4]
                budget = J - j
                per_color = []
                dead = False
                for c in range(4):
                    a = avail[(c, n)]
                    tb = mand[(c, n)]
                    opts = []
                    for u in range(tb, a + 1):
                        for x_real in range(0, u + 1):
                            g = u - x_real
                            if g > 2:
                                continue
                            for rj in range(0, budget + 1):
                                trans = RUN_TRANS.get((state[c], x_real + rj))
                                if not trans:
                                    continue
                                for (npair, extl, n_new) in trans:
                                    opts.append((npair, g, u - tb, rj,
                                                 (x_real, rj, extl, n_new)))
                    if not opts:
                        dead = True
                        break
                    per_color.append(opts)
                if dead:
                    continue

                for o0 in per_color[0]:
                    for o1 in per_color[1]:
                        for o2 in per_color[2]:
                            for o3 in per_color[3]:
                                gvec = (o0[1], o1[1], o2[1], o3[1])
                                run_jok = o0[3] + o1[3] + o2[3] + o3[3]
                                if j + run_jok > J:
                                    continue
                                nstate_runs = (o0[0], o1[0], o2[0], o3[0])
                                gain = o0[2] + o1[2] + o2[2] + o3[2]
                                decisions = (o0[4], o1[4], o2[4], o3[4])
                                for jg in range(0, J - j - run_jok + 1):
                                    if any(gvec) or jg:
                                        witness = GROUP_PARTS_J.get((*gvec, jg))
                                        if witness is None:
                                            continue
                                    else:
                                        witness = []
                                    nj = j + run_jok + jg
                                    nstate = nstate_runs + (nj,)
                                    tgt = new_layer.setdefault(nstate, {})
                                    for tiles in keymap:
                                        nt = tiles + gain
                                        if nt not in tgt:
                                            tgt[nt] = True
                                            layer_parents[(nstate, nt)] = (
                                                state, tiles, decisions,
                                                gvec, jg, witness,
                                            )
            parents.append(layer_parents)
            layer = new_layer
            if not layer:
                return None
        return layer, parents, J

    def _traceback_joker(self, state, key, parents):
        chain = [None] * 13
        s, k = state, key
        for n in range(13, 0, -1):
            rec = parents[n - 1][(s, k)]
            prev_state, prev_key, decisions, gvec, jg, witness = rec
            chain[n - 1] = (decisions, gvec, jg, witness)
            s, k = prev_state, prev_key

        # runs are lists of concrete tiles (Tile or JOKER), in number order
        sets = []
        open_runs = [[] for _ in range(4)]
        for n in range(1, 14):
            decisions, gvec, jg, witness = chain[n - 1]
            for c in range(4):
                x_real, rj, extl, n_new = decisions[c]
                remaining = list(open_runs[c])
                kept = []
                # tiles placed at (c, n) into runs: x_real reals + rj jokers
                slot_tiles = [Tile(COLORS[c], n)] * x_real + [JOKER] * rj
                si = 0
                for capped in extl:
                    for i, run in enumerate(remaining):
                        if min(len(run), 3) == capped:
                            run.append(slot_tiles[si]); si += 1
                            kept.append(run)
                            del remaining[i]
                            break
                for run in remaining:            # runs not extended -> close
                    sets.append(run)
                for _ in range(n_new):
                    kept.append([slot_tiles[si]]); si += 1
                open_runs[c] = kept
            for cols, njok in witness:
                sets.append([Tile(COLORS[c], n) for c in sorted(cols)]
                            + [JOKER] * njok)
        for c in range(4):
            for run in open_runs[c]:
                sets.append(run)
        return sets

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
