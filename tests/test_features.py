import numpy as np

from asl.features import FEATURE_DIM, flatten_features, normalize_landmarks


def _hand(shift=0.0, scale=1.0) -> np.ndarray:
    template = np.zeros((21, 3), dtype=np.float32)
    template[:, 0] = np.linspace(0, 1, 21)
    template[9] = [0.4, 0.1, 0.0]
    return template * scale + np.array([shift, shift, 0.0], dtype=np.float32)


def test_wrist_is_origin():
    out = normalize_landmarks(_hand(shift=0.3, scale=2.0))
    assert np.allclose(out[0], 0.0, atol=1e-6)


def test_scale_invariant():
    a = flatten_features(_hand(shift=0.0, scale=1.0))
    b = flatten_features(_hand(shift=0.2, scale=3.0))
    assert a.shape == (FEATURE_DIM,)
    assert np.allclose(a, b, atol=1e-5)


def test_rejects_wrong_shape():
    try:
        normalize_landmarks(np.zeros((10, 3)))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
