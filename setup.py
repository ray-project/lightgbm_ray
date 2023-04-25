from setuptools import find_packages, setup

setup(
    name="lightgbm_ray",
    packages=find_packages(where=".", include="lightgbm_ray*"),
    version="0.1.9",
    author="Ray Team",
    description="A Ray backend for distributed LightGBM",
    license="Apache 2.0",
    long_description="A distributed backend for LightGBM built on top of "
    "distributed computing framework Ray.",
    url="https://github.com/ray-project/lightgbm_ray",
    install_requires=["lightgbm>=3.2.1", "xgboost_ray>=0.1.12", "packaging"],
)
