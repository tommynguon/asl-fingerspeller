import numpy as np
import pytest

from asl.collect import append_row
from asl.features import FEATURE_DIM


def test_append_row_validates_label_and_shape(tmp_path):
    path = tmp_path / "landmarks.csv"
    append_row("a", "test", np.zeros(FEATURE_DIM), path=path)
    assert path.read_text(encoding="utf-8").splitlines()[1].startswith("A,test,")

    with pytest.raises(ValueError, match="unknown label"):
        append_row("invalid", "test", np.zeros(FEATURE_DIM), path=path)
    with pytest.raises(ValueError, match="expected 63 features"):
        append_row("A", "test", np.zeros(10), path=path)
