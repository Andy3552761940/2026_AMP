import numpy as np
import pandas as pd

from hemo_pred.features import aac_features, physchem_features, build_handcrafted_matrix


def test_aac_sum_one_for_nonempty():
    v = aac_features("AKLV")
    assert np.isclose(v.sum(), 1.0)


def test_physchem_shape():
    v = physchem_features("AKLV")
    assert v.shape == (5,)


def test_build_matrix_shape():
    df = pd.DataFrame({"sequence": ["AKLV", "DDD"]})
    X = build_handcrafted_matrix(df)
    assert X.shape == (2, 25)
