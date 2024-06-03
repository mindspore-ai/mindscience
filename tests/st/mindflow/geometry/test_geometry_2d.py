"""test geometry module: 2d cases"""
import pytest
from easydict import EasyDict as edict

from mindflow.geometry import generate_sampling_config
from mindflow.geometry import Triangle

triangle_random = edict(
    {
        "domain": edict({"random_sampling": True, "size": 100, "sampler": "uniform"}),
        "BC": edict(
            {
                "random_sampling": True,
                "size": 100,
                "sampler": "uniform",
            }
        ),
    }
)


def check_triangle_case(triangle_config):
    """check_triangle_case"""
    with pytest.raises(ValueError):
        Triangle(
            "triangle",
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            sampling_config=generate_sampling_config(triangle_config),
        )

    triangle = Triangle(
        "triangle",
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        sampling_config=generate_sampling_config(triangle_config),
    )
    domain = triangle.sampling(geom_type="domain")
    bc = triangle.sampling(geom_type="BC")
    with pytest.raises(ValueError):
        triangle.sampling(geom_type="other")

    print(domain, bc)
    assert len(domain) == 100
    assert len(bc) == 100


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_check_triangle_case():
    """
    Feature: check triangle sampling
    Description: None.
    Expectation: Success or raise AssertionError when number of sampling points is not correct.
    """
    check_triangle_case(triangle_random)
