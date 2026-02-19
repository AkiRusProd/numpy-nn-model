import os
import sys

import numpy as np
import pytest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neunet as nnet


def _cuda_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.parametrize("factory,args", [
    (nnet.ones, (2, 3)),
    (nnet.zeros, (2, 3)),
    (nnet.rand, (2, 3)),
    (nnet.randn, (2, 3)),
    (nnet.arange, (5,)),
])
def test_creation_wrappers_cpu(factory, args):
    x = factory(*args, device="cpu")
    assert x.device == "cpu"
    assert x.dtype == np.float32


def test_creation_wrappers_cpu_dtype():
    x1 = nnet.ones(2, 3, device="cpu", dtype=np.float64)
    x2 = nnet.arange(5, device="cpu", dtype=np.int32)

    assert x1.dtype == np.float64
    assert x2.dtype == np.int32


def test_shape_normalization():
    a = nnet.ones((2, 3), device="cpu")
    b = nnet.zeros([2, 3], device="cpu")
    assert a.shape == (2, 3)
    assert b.shape == (2, 3)


def test_like_and_arg_wrappers_cpu():
    x = nnet.tensor([[1.0, 3.0, 2.0]], device="cpu")
    ones = nnet.ones_like(x)
    zeros = nnet.zeros_like(x)

    assert ones.device == "cpu"
    assert zeros.device == "cpu"
    assert ones.shape == x.shape
    assert zeros.shape == x.shape

    argmax = nnet.argmax(x, axis=1)
    argmin = nnet.argmin(x, axis=1)
    assert argmax.device == "cpu"
    assert argmin.device == "cpu"
    assert argmax.dtype == np.int32
    assert argmin.dtype == np.int32
    assert int(argmax.data[0]) == 1
    assert int(argmin.data[0]) == 0


def test_rand_randn_dtype_none_regression():
    x = nnet.rand(2, 3, device="cpu")
    y = nnet.randn(2, 3, device="cpu")
    assert x.dtype == np.float32
    assert y.dtype == np.float32


def test_save_load_roundtrip(tmp_path):
    obj = {"a": 1, "b": [1, 2, 3]}
    path = tmp_path / "obj.nt"
    nnet.save(obj, path)
    loaded = nnet.load(path)
    assert loaded == obj


@pytest.mark.skipif(not _cuda_available(), reason="CUDA is not available")
def test_creation_wrappers_cuda():
    x1 = nnet.ones(2, 3, device="cuda")
    x2 = nnet.zeros(2, 3, device="cuda")
    x3 = nnet.rand(2, 3, device="cuda")
    x4 = nnet.randn(2, 3, device="cuda")
    x5 = nnet.arange(5, device="cuda")

    assert x1.device == "cuda"
    assert x2.device == "cuda"
    assert x3.device == "cuda"
    assert x4.device == "cuda"
    assert x5.device == "cuda"


@pytest.mark.skipif(not _cuda_available(), reason="CUDA is not available")
def test_like_and_arg_wrappers_cuda():
    x = nnet.tensor([[1.0, 3.0, 2.0]], device="cuda")
    ones = nnet.ones_like(x)
    zeros = nnet.zeros_like(x)
    assert ones.device == "cuda"
    assert zeros.device == "cuda"

    argmax = nnet.argmax(x, axis=1)
    argmin = nnet.argmin(x, axis=1)
    assert argmax.device == "cuda"
    assert argmin.device == "cuda"
    assert argmax.dtype == np.int32
    assert argmin.dtype == np.int32

