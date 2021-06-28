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
        evals=[(train_set, "train")],
        verbose=False,
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


if __name__ == "__main__":
    import ray

    ray.init(num_cpus=5)

    print("Readme: Simple example")
    readme_simple()
    readme_predict()
