import argparse

import numpy as np
import pandas as pd
import ray
from packaging.version import Version
from sklearn.utils import shuffle
from xgboost_ray.data_sources.modin import MODIN_INSTALLED

from lightgbm_ray import RayDMatrix, RayParams, train


def main(cpus_per_actor, num_actors):
    if not MODIN_INSTALLED:
        print(
            "Modin is not installed or installed in a version that is not "
            "compatible with lightgbm_ray (< 0.9.0)."
        )
        return

    import modin

    if Version(modin.__version__) < Version("0.16.0") and Version(
        ray.__version__
    ) >= Version("2.6.0"):
        print("modin<=0.16.0 is not compatible with ray>=2.6.0.")
        return

    # Import modin after initializing Ray
    from modin.distributed.dataframe.pandas import from_partitions

    # Generate dataset
    x = np.repeat(range(8), 16).reshape((32, 4))
    # Even numbers --> 0, odd numbers --> 1
    y = np.tile(np.repeat(range(2), 4), 4)

    # Flip some bits to reduce max accuracy
    bits_to_flip = np.random.choice(32, size=6, replace=False)
    y[bits_to_flip] = 1 - y[bits_to_flip]

    # LightGBM requires well-shuffled data
    x, y = shuffle(x, y, random_state=1)

    data = pd.DataFrame(x)
    data["label"] = y

    # Split into 4 partitions
    partitions = [ray.put(part) for part in np.split(data, 4)]

    # Create modin df here
    modin_df = from_partitions(partitions, axis=0)

    train_set = RayDMatrix(modin_df, "label")

    evals_result = {}
    # Set LGBM config.
    lightgbm_params = {
        "objective": "binary",
        "metric": ["binary_logloss", "binary_error"],
    }

    # Train the classifier
    bst = train(
        params=lightgbm_params,
        dtrain=train_set,
        valid_sets=[train_set],
        valid_names=["train"],
        evals_result=evals_result,
        ray_params=RayParams(
            max_actor_restarts=0,
            gpus_per_actor=0,
            cpus_per_actor=cpus_per_actor,
            num_actors=num_actors,
        ),
        verbose_eval=False,
        num_boost_round=10,
    )

    model_path = "modin.lgbm"
    bst.booster_.save_model(model_path)
    print(
        "Final training error: {:.4f}".format(evals_result["train"]["binary_error"][-1])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address", required=False, type=str, help="the address to use for Ray"
    )
    parser.add_argument(
        "--server-address",
        required=False,
        type=str,
        help="Address of the remote server if using Ray Client.",
    )
    parser.add_argument(
        "--cpus-per-actor",
        type=int,
        default=2,
        help="Sets number of CPUs per lightgbm training worker.",
    )
    parser.add_argument(
        "--num-actors",
        type=int,
        default=2,
        help="Sets number of lightgbm workers to use.",
    )
    parser.add_argument("--smoke-test", action="store_true", default=False, help="gpu")

    args, _ = parser.parse_known_args()

    if args.smoke_test:
        ray.init(num_cpus=(args.num_actors * args.cpus_per_actor) + 1)
    elif args.server_address:
        ray.util.connect(args.server_address)
    else:
        ray.init(address=args.address)

    main(args.cpus_per_actor, args.num_actors)
