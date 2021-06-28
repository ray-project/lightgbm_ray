import os

from sklearn import datasets

import lightgbm as lgbm
from lightgbm_ray import RayDMatrix, predict

import numpy as np


def main():
    if not os.path.exists("simple.lgbm"):
        raise ValueError("Model file not found: `simple.lgbm`"
                         "\nFIX THIS by running `python `simple.py` first to "
                         "train the model.")

    # Load dataset
    data, labels = datasets.load_breast_cancer(return_X_y=True)

    dmat_ray = RayDMatrix(data, labels)

    bst = lgbm.Booster(model_file="simple.lgbm")

    pred_lgbm = bst.predict(data)
    pred_ray = predict(bst, dmat_ray)

    np.testing.assert_array_equal(pred_lgbm, pred_ray)
    print(pred_ray)


if __name__ == "__main__":
    main()
