from test_end_to_end import LGBMRayEndToEndTest


class LGBMRayEndToEndTestVoting(LGBMRayEndToEndTest):
    def setUp(self):
        super().setUp()
        self.params["tree_learner"] = "voting"


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
