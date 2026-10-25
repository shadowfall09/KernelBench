"""Shared CPU-only test environment; device/model experiments are separate."""

import pytest


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
