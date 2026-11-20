import numpy as np

from src.gesture_runtime import landmark_feature_vector


def test_landmark_feature_vector_is_finite_and_stable():
    xy = np.arange(42, dtype=np.float32).reshape(21, 2) / 20
    z = np.linspace(-.2, .2, 21, dtype=np.float32).reshape(21, 1)
    vector = landmark_feature_vector(xy, z)
    # 42 xy + 21 z + 20 wrist radii + 20 bones + 10 fingertip gaps.
    assert vector.shape == (113,)
    assert np.isfinite(vector).all()
    np.testing.assert_allclose(vector, landmark_feature_vector(xy, z))
