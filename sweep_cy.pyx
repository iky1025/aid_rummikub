# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython-compiled jokered DP layer sweep — the rollout hotspot (_sweep_joker).

Behaviour-identical to rummikub_dp._sweep_joker (verified by crosscheck). The
layer/state structures stay Python dicts/tuples (dynamic + needed for
traceback); the win is C-typed loop counters + C arrays for avail/mand + a
per-color opts memo, all compiled to C. RUN_TRANS/GROUP_PARTS_J/EMPTY_PAIR are
passed in to avoid a circular import with rummikub_dp.
"""
from rummikub_solver import COLORS, Tile

# Flat GROUP_PARTS_J: witness indexed by g0+3*g1+9*g2+27*g3+81*jg (g in 0..2,
# jg small) -> replaces the per-iteration 5-tuple build + dict.get in the hot
# loop with a C-int index + list access. Cached by the dict's identity.
_gpj_src = None
_gpj_arr = None


cdef _get_gpj_arr(GROUP_PARTS_J):
    global _gpj_src, _gpj_arr
    if _gpj_src is GROUP_PARTS_J:
        return _gpj_arr
    arr = [None] * (81 * 32)
    for key, val in GROUP_PARTS_J.items():
        g0, g1, g2, g3, jg = key
        idx = g0 + 3 * g1 + 9 * g2 + 27 * g3 + 81 * jg
        if 0 <= idx < len(arr):
            arr[idx] = val
    _gpj_src = GROUP_PARTS_J
    _gpj_arr = arr
    return arr


def sweep_joker(hand_counter, table_counter, int jt, int jh,
                RUN_TRANS, GROUP_PARTS_J, EMPTY_PAIR):
    cdef int J = jt + jh
    cdef int c, n, u, x_real, g, rj, jg
    cdef int j, budget, tb, a, utb, run_jok, nj0, nj, gain, tiles, nt
    cdef int o0g, o1g, o2g, o3g, o0t, o1t, o2t, o3t, o0j, o1j, o2j, o3j, idx
    cdef int avail[4][14]
    cdef int mand[4][14]
    cdef int tv, hv
    gpj_arr = _get_gpj_arr(GROUP_PARTS_J)

    for c in range(4):
        for n in range(1, 14):
            tile = Tile(COLORS[c], n)
            tv = table_counter.get(tile, 0)
            hv = hand_counter.get(tile, 0)
            mand[c][n] = tv
            avail[c][n] = tv + hv

    start_state = (EMPTY_PAIR,) * 4 + (0,)
    layer = {start_state: {0: None}}
    parents = []
    RT = RUN_TRANS
    GPJ = GROUP_PARTS_J

    for n in range(1, 14):
        new_layer = {}
        layer_parents = {}
        copts_memo = {}                      # (c, state_c, budget) -> opts
        for state, keymap in layer.items():
            j = state[4]
            budget = J - j
            per_color = []
            dead = 0
            for c in range(4):
                sc = state[c]
                mkey = (c, sc, budget)
                opts = copts_memo.get(mkey)
                if opts is None:
                    a = avail[c][n]
                    tb = mand[c][n]
                    opts = []
                    for u in range(tb, a + 1):
                        utb = u - tb
                        for x_real in range(0, u + 1):
                            g = u - x_real
                            if g > 2:
                                continue
                            for rj in range(0, budget + 1):
                                trans = RT.get((sc, x_real + rj))
                                if not trans:
                                    continue
                                for (npair, extl, n_new) in trans:
                                    opts.append((npair, g, utb, rj,
                                                 (x_real, rj, extl, n_new)))
                    copts_memo[mkey] = opts
                if not opts:
                    dead = 1
                    break
                per_color.append(opts)
            if dead:
                continue

            p0 = per_color[0]; p1 = per_color[1]
            p2 = per_color[2]; p3 = per_color[3]
            for o0 in p0:
                o0p = o0[0]; o0g = o0[1]; o0t = o0[2]; o0j = o0[3]; o0d = o0[4]
                for o1 in p1:
                    o1p = o1[0]; o1g = o1[1]; o1t = o1[2]; o1j = o1[3]; o1d = o1[4]
                    for o2 in p2:
                        o2p = o2[0]; o2g = o2[1]; o2t = o2[2]; o2j = o2[3]; o2d = o2[4]
                        for o3 in p3:
                            o3g = o3[1]; o3j = o3[3]
                            run_jok = o0j + o1j + o2j + o3j
                            nj0 = j + run_jok
                            if nj0 > J:
                                continue
                            gvec = (o0g, o1g, o2g, o3g)
                            gain = o0t + o1t + o2t + o3[2]
                            nstate_runs = (o0p, o1p, o2p, o3[0])
                            decisions = (o0d, o1d, o2d, o3[4])
                            idx = o0g + 3 * o1g + 9 * o2g + 27 * o3g
                            for jg in range(0, J - nj0 + 1):
                                if idx or jg:
                                    witness = gpj_arr[idx + 81 * jg]
                                    if witness is None:
                                        continue
                                else:
                                    witness = []
                                nj = nj0 + jg
                                nstate = nstate_runs + (nj,)
                                tgt = new_layer.get(nstate)
                                if tgt is None:
                                    tgt = {}
                                    new_layer[nstate] = tgt
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
