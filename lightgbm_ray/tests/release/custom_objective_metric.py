import ray

from lightgbm_ray.tests.test_lightgbm_api import LightGBMAPITest


class LightGBMDistributedAPITest(LightGBMAPITest):
    def _init_ray(self):
        if not ray.is_initialized():
            ray.init(address="auto")


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", f"{__file__}::LightGBMDistributedAPITest"]))
