import ctypes
from ctypes import POINTER, c_float, c_size_t
import cupy as cp
import neunet.nn as nn
from neunet.autograd import Tensor
from neunet.nn.experimental.utils import (
    CUDA_SOFTMAX_MODULE,
    get_module_path,
    load_dlls,
    load_cuda_function,
    call_cuda_function,
    get_current_stream_ptr,
)

load_dlls()

CUDA_SOFTMAX_DLL = get_module_path(CUDA_SOFTMAX_MODULE)

CUDA_SOFTMAX_FORWARD = load_cuda_function(
    CUDA_SOFTMAX_DLL, "cudaSoftmaxForward", 
    [
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        c_size_t, 
        c_size_t, 
        c_size_t,
        ctypes.c_void_p
    ]
)

CUDA_SOFTMAX_BACKWARD = load_cuda_function(
    CUDA_SOFTMAX_DLL, "cudaSoftmaxBackward", 
    [
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        c_size_t,
        c_size_t,
        c_size_t,
        ctypes.c_void_p
    ]
)



# class ndarray:
#     def __init__(self, array: Union[np.ndarray, cp.ndarray]):
#         self.array = array

#     def __array__(self):
#         return self.array

def cuda_softmax_forward(x: cp.ndarray, o: cp.ndarray, dim: int):
    if not all([isinstance(arg, cp.ndarray) for arg in [x, o]]):
        raise ValueError("All arguments must be cupy arrays.")
    if x.shape != o.shape:
        raise ValueError("Input and output shapes must match")
    
    if not x.flags.c_contiguous:
        x = cp.ascontiguousarray(x)
    if not o.flags.c_contiguous:
        o = cp.ascontiguousarray(o)
    
    shape = x.shape
    ndim = x.ndim

    dim = dim % ndim 

    # Explanation:
    # Let's say the tensor has shape (2, 3, 4), and dim=1 (axis of size 3). Then:
    # slice_size = 3 (size of dim axis).
    # num_slices = (2 * 4) = 8 (number of slices).
    # Each block processes one vector of 3 elements.
    # Each thread processes 1 element at 1 moment.

    slice_size = shape[dim]
    num_slices = x.size // slice_size

    # stride = 1
    # for i in range(dim + 1, ndim):
    #     stride *= shape[i]

    stride = x.strides[dim] // x.itemsize

    stream_ptr = get_current_stream_ptr()
    
    call_cuda_function(
        CUDA_SOFTMAX_FORWARD, 
        o, 
        x, 
        num_slices,
        slice_size,
        stride,
        stream_ptr
    )
    return o

def cuda_softmax_backward(grad_x: cp.ndarray, grad: cp.ndarray, f_x: cp.ndarray, dim: int):
    if not all([isinstance(arg, cp.ndarray) for arg in [grad_x, grad, f_x]]):
        raise ValueError("All arguments must be cupy arrays.")
    if grad_x.shape != grad.shape and grad_x.shape != f_x.shape:
        raise ValueError("Input, output gradients and softmax shapes must match")
    
    if not grad.flags.c_contiguous:
        grad = cp.ascontiguousarray(grad)
    if not f_x.flags.c_contiguous:
        f_x = cp.ascontiguousarray(f_x)

    shape = grad.shape
    ndim = grad.ndim

    dim = dim % ndim 
    
    slice_size = shape[dim]
    num_slices = grad.size // slice_size
    
    # stride = 1
    # for i in range(dim + 1, ndim):
    #     stride *= shape[i]

    stride = f_x.strides[dim] // f_x.itemsize
    
    stream_ptr = get_current_stream_ptr()
    
    call_cuda_function(
        CUDA_SOFTMAX_BACKWARD,
        grad_x,
        grad,
        f_x,
        num_slices,
        slice_size,
        stride,
        stream_ptr
    )
    return grad_x


class _CUDASoftmaxTensor(Tensor):
    def __init__(self, data, args, op, device):
        """Fused softmax with CUDA backend."""
        super().__init__(data, args, op, device=device)

        def grad_fn(t: Tensor, f_x, axis, grad):
            if not all(arr.dtype == "float32" for arr in [t, f_x, grad]):
                raise NotImplementedError(f"Only float32 is supported.")
            
            grad_x = t.xp.empty_like(t.data)

            cuda_softmax_backward(grad_x, grad, f_x, axis)

            t.apply_grad(grad_x)

        self.grad_fn = grad_fn

class CUDASoftmax(nn.Module):
    def __init__(self, axis: int = 1):
        super(CUDASoftmax, self).__init__()
        self.axis = axis

    def forward(self, x: Tensor):
        if x.dtype != "float32":
            raise NotImplementedError(f"Only float32 is supported, got {x.dtype} instead.")
        if x.device != "cuda":
            raise NotImplementedError(f"Only CUDA is supported with cupy backend, got {x.device} instead.")
        
        f_x = x.xp.empty_like(x.data)
        cuda_softmax_forward(x.data, f_x, self.axis)
        
        return _CUDASoftmaxTensor(f_x, [x, f_x, self.axis], "softmax", device=x.device)
