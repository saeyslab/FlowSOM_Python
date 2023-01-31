import pytest

import FlowSOM


def test_package_has_version():
    FlowSOM.__version__


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_example():
    assert 1 == 1  # This test is designed to fail.
