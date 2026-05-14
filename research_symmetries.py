import numpy as np
import stim

def get_detector_coords(d, rounds=None):
    if rounds is None: rounds = d
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=rounds)
    raw = circuit.get_detector_coordinates()
    coords = np.zeros((circuit.num_detectors, 3), dtype=np.float32)
    for idx, c in raw.items():
        # Defensive: Stim can emit <3-d coords for some boundary detectors.
        c3 = np.asarray(c[:3], dtype=np.float32)
        coords[idx, :c3.shape[0]] = c3
    return coords

def get_rot180_permutation(coords):
    x_mid, y_mid = (coords[:, 0].min() + coords[:, 0].max()) / 2, (coords[:, 1].min() + coords[:, 1].max()) / 2
    perm = np.zeros(len(coords), dtype=np.int32)
    for i in range(len(coords)):
        target = [2*x_mid - coords[i, 0], 2*y_mid - coords[i, 1], coords[i, 2]]
        dists = np.linalg.norm(coords - target, axis=1)
        perm[i] = np.argmin(dists)
    return perm

def get_time_reversal_permutation(coords):
    # Reverse rounds: t -> (max_t - t)
    t_min, t_max = coords[:, 2].min(), coords[:, 2].max()
    perm = np.zeros(len(coords), dtype=np.int32)
    for i in range(len(coords)):
        target = [coords[i, 0], coords[i, 1], t_max - (coords[i, 2] - t_min)]
        dists = np.linalg.norm(coords - target, axis=1)
        perm[i] = np.argmin(dists)
    return perm


def _make_perm(coords, transform_fn, snap_tol=0.5):
    """Build nearest-neighbor permutation for transform_fn: (x,y,t) -> (x',y',t').

    Returns None if any source point has no detector within snap_tol of its
    target, or if the resulting index map isn't bijective (i.e. the transform
    isn't a valid detector-to-detector symmetry of the lattice).
    """
    perm = np.zeros(len(coords), dtype=np.int32)
    for i in range(len(coords)):
        target = np.asarray(transform_fn(coords[i]), dtype=coords.dtype)
        dists = np.linalg.norm(coords - target, axis=1)
        j = int(np.argmin(dists))
        if dists[j] > snap_tol:
            return None
        perm[i] = j
    if len(np.unique(perm)) != len(perm):
        return None
    return perm


def get_d4_permutations(coords, include_time_reversal=True):
    """All D4 spatial syms (and optional time reversal compositions) that
    are valid bijective detector permutations.

    Returns dict {name: perm_array}. Invalid candidates (e.g. 90 deg
    rotations of the rotated_memory_z lattice, where X/Z stabilizers swap
    so the transform is not a detector-to-detector map) are filtered out.
    """
    cx = (coords[:, 0].min() + coords[:, 0].max()) / 2
    cy = (coords[:, 1].min() + coords[:, 1].max()) / 2
    tmin, tmax = coords[:, 2].min(), coords[:, 2].max()

    def t_fwd(t): return t
    def t_rev(t): return tmax - (t - tmin)

    spatial = {
        'e':  lambda c: [c[0],                  c[1],                  c[2]],
        'r':  lambda c: [cx - (c[1] - cy),      cy + (c[0] - cx),      c[2]],
        'r2': lambda c: [2*cx - c[0],           2*cy - c[1],           c[2]],
        'r3': lambda c: [cx + (c[1] - cy),      cy - (c[0] - cx),      c[2]],
        'sx': lambda c: [2*cx - c[0],           c[1],                  c[2]],
        'sy': lambda c: [c[0],                  2*cy - c[1],           c[2]],
        'sd': lambda c: [cx + (c[1] - cy),      cy + (c[0] - cx),      c[2]],
        'sa': lambda c: [cx - (c[1] - cy),      cy - (c[0] - cx),      c[2]],
    }
    time_options = [('', t_fwd)]
    if include_time_reversal:
        time_options.append(('T', t_rev))

    out = {}
    for sname, sfn in spatial.items():
        for tname, tfn in time_options:
            full = lambda c, s=sfn, tf=tfn: [*s(c)[:2], tf(s(c)[2])]
            perm = _make_perm(coords, full)
            if perm is not None:
                out[sname + tname] = perm
    return out


if __name__ == "__main__":
    # Verify for d=7
    d = 7
    coords = get_detector_coords(d)
    p_rot = get_rot180_permutation(coords)
    p_time = get_time_reversal_permutation(coords)
    print(f"Distance {d}")
    print(f"Rot 180 is valid permutation: {len(np.unique(p_rot)) == len(p_rot)}")
    print(f"Time Reversal is valid permutation: {len(np.unique(p_time)) == len(p_time)}")
    d4 = get_d4_permutations(coords)
    print(f"D4 x T valid permutations ({len(d4)}): {list(d4)}")
