# flake8: noqa E501


def readme_simple():
    from lightgbm_ray import RayDMatrix, RayParams, train
    from sklearn.datasets import load_breast_cancer

    train_x, train_y = load_breast_cancer(return_X_y=True)
    train_set = RayDMatrix(train_x, train_y)

    evals_result = {}
    bst = train(
        {
            "objective": "binary",
            "metric": ["binary_logloss", "binary_error"],
        },
        train_set,
        evals_result=evals_result,
        valid_sets=[train_set],
        valid_names=["train"],
        verbose_eval=False,
        ray_params=RayParams(num_actors=2, cpus_per_actor=2))

    bst.booster_.save_model("model.lgbm")
    print("Final training error: {:.4f}".format(
        evals_result["train"]["binary_error"][-1]))


def readme_predict():
    from lightgbm_ray import RayDMatrix, RayParams, predict
    from sklearn.datasets import load_breast_cancer
    import lightgbm as lgbm

    data, labels = load_breast_cancer(return_X_y=True)

    dpred = RayDMatrix(data, labels)

    bst = lgbm.Booster(model_file="model.lgbm")
    pred_ray = predict(bst, dpred, ray_params=RayParams(num_actors=2))

    print(pred_ray)


def readme_tune():
    from lightgbm_ray import RayDMatrix, RayParams, train
    from sklearn.datasets import load_breast_cancer

    num_actors = 2
    num_cpus_per_actor = 2

    ray_params = RayParams(
        num_actors=num_actors, cpus_per_actor=num_cpus_per_actor)

    def train_model(config):
        train_x, train_y = load_breast_cancer(return_X_y=True)
        train_set = RayDMatrix(train_x, train_y)

        evals_result = {}
        bst = train(
            params=config,
            dtrain=train_set,
            evals_result=evals_result,
            valid_sets=[train_set],
            valid_names=["train"],
            verbose_eval=False,
            ray_params=ray_params)
        bst.booster_.save_model("model.lgbm")

    from ray import tune

    # Specify the hyperparameter search space.
    config = {
        "objective": "binary",
        "metric": ["binary_logloss", "binary_error"],
        "eta": tune.loguniform(1e-4, 1e-1),
        "subsample": tune.uniform(0.5, 1.0),
        "max_depth": tune.randint(1, 9)
    }

    # Make sure to use the `get_tune_resources` method to set the `resources_per_trial`
    analysis = tune.run(
        train_model,
        config=config,
        metric="train-binary_error",
        mode="min",
        num_samples=4,
        resources_per_trial=ray_params.get_tune_resources())
    print("Best hyperparameters", analysis.best_config)


if __name__ == "__main__":
    import ray

    ray.init(num_cpus=5)

    print("Readme: Simple example")
    readme_simple()
    readme_predict()
    try:
        print("Readme: Ray Tune example")
        readme_tune()
    except ImportError:
        print("Ray Tune not installed.")