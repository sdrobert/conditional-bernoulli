import pytest
import torch
import zlib


@pytest.fixture(scope="function")
def global_seed(request):
    torch.manual_seed(abs(zlib.adler32(bytes(request.function.__name__, "utf-8"))))


@pytest.fixture(
    scope="session",
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda is not available"
            ),
        ),
    ],
)
def device(request):
    return torch.device(request.param)
