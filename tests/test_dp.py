import numpy as np

from dicom2waveform.envelope.dp import _l1_min_convolution, dp_solve, edge_cost, split_cost


def test_l1_min_convolution_matches_brute_force():
    rng = np.random.default_rng(0)
    for lam in (0.0, 0.3, 2.0):
        f = rng.random(40) * 10
        h, arg = _l1_min_convolution(f, lam)
        i = np.arange(40)
        brute = f[None, :] + lam * np.abs(i[:, None] - i[None, :])
        assert np.allclose(h, brute.min(axis=1))
        assert np.allclose(brute[i, arg], h)


def test_dp_solve_matches_brute_force_small():
    rng = np.random.default_rng(1)
    D, W, lam = 4, 5, 0.7
    C = rng.random((D, W))
    path = dp_solve(C, lam)
    best = np.inf
    for code in range(D ** W):
        p = [(code // D ** j) % D for j in range(W)]
        c = sum(C[p[j], j] for j in range(W)) + lam * sum(abs(p[j] - p[j - 1]) for j in range(1, W))
        best = min(best, c)
    got = sum(C[path[j], j] for j in range(W)) + lam * np.abs(np.diff(path)).sum()
    assert np.isclose(got, best)


def test_edge_cost_no_flow_gives_zero_and_band_gives_edge():
    noise = np.full((100, 3), 0.1)
    assert np.all(np.argmin(edge_cost(noise, theta=0.5, inner_rows=5), axis=0) == 0)
    band = noise.copy()
    band[40:60] = 0.9  # hollow band: interior rows 0..39 hold no signal
    assert np.all(np.argmin(edge_cost(band, theta=0.5, inner_rows=5), axis=0) == 59)
    # the MATLAB split cost assumes a filled region and collapses on a hollow band
    assert np.all(np.argmin(split_cost(band), axis=0) < 40)
