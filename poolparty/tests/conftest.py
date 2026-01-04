"""Pytest configuration for poolparty tests."""
import pytest
import poolparty as pp


@pytest.fixture(autouse=True)
def reset_party_before_each_test():
    """Reset the default party before each test to ensure isolation."""
    pp.init()
    yield
