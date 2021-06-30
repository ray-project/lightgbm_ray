import argparse

from sklearn import datasets
from sklearn.model_selection import train_test_split

import ray

from lightgbm_ray import RayDMatrix, train, RayParams


def main(cpus_per_actor, num_actors):
    # Load dataset
    data, labels = datasets.load_breast_cancer(return_X_y=True)
    # Split into train and test set
    train_x, test_x, train_y, test_y = train_test_split(
        data, labels, test_size=0.25)

    train_set = RayDMatrix(train_x, train_y)
    test_set = RayDMatrix(test_x, test_y)

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
        valid_sets=[test_set],
        valid_names=["eval"],
        evals_result=evals_result,
        ray_params=RayParams(
            max_actor_restarts=0,
            gpus_per_actor=0,
            cpus_per_actor=cpus_per_actor,
            num_actors=num_actors),
        verbose_eval=False,
        num_boost_round=10)

    model_path = "simple.lgbm"
    bst.booster_.save_model(model_path)
    print("Final validation error: {:.4f}".format(
        evals_result["eval"]["binary_error"][-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        required=False,
        type=str,
        help="the address to use for Ray")
    parser.add_argument(
        "--server-address",
        required=False,
        type=str,
        help="Address of the remote server if using Ray Client.")
    parser.add_argument(
        "--cpus-per-actor",
        type=int,
        default=2,
        help="Sets number of CPUs per lightgbm training worker.")
    parser.add_argument(
        "--num-actors",
        type=int,
        default=2,
        help="Sets number of lightgbm workers to use.")
    parser.add_argument(
        "--smoke-test", action="store_true", default=False, help="gpu")

    args, _ = parser.parse_known_args()

    if args.smoke_test:
        ray.init(num_cpus=args.num_actors * args.cpus_per_actor)
    elif args.server_address:
        ray.util.connect(args.server_address)
    else:
        ray.init(address=args.address)

    main(args.cpus_per_actor, args.num_actors)
