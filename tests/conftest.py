import random

import pytest


@pytest.fixture
def rng():
    return random.Random(20261027)
