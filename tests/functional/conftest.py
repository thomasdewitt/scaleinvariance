import numpy as np
import pytest


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
    except ImportError:
        pass
