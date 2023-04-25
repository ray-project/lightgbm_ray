import os

import lightgbm as lgbm
import numpy as np
from sklearn import datasets

from lightgbm_ray import RayDMatrix, RayParams, predict


def main():
    if not os.path.exists("simple.lgbm"):
        raise ValueError(
            "Model file not found: `simple.lgbm`"
            "\nFIX THIS by running `python `simple.py` first to "
            "train the model."
        )

    # Load dataset
    data, labels = datasets.load_breast_cancer(return_X_y=True)

    dmat_ray = RayDMatrix(data, labels)

    bst = lgbm.Booster(model_file="simple.lgbm")

    pred_lgbm = bst.predict(data)
    pred_ray = predict(bst, dmat_ray, ray_params=RayParams(num_actors=2))

    np.testing.assert_array_equal(pred_lgbm, pred_ray)
    print(pred_ray)


if __name__ == "__main__":
    main()
