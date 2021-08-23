import test_end_to_end


class LGBMRayEndToEndTestVoting(test_end_to_end.LGBMRayEndToEndTest):
    def setUp(self):
        super().setUp()
        self.params["tree_learner"] = "voting"


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
