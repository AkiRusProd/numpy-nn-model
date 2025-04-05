import ctypes
from ctypes import POINTER, c_float, c_int, c_char
import cupy as cp

import neunet.nn as nn
from neunet.autograd import Tensor
from neunet.nn.experimental.utils import CUDA_CROSS_ENTROPY_MODULE, get_module_path, load_dlls

load_dlls()

CUDA_CROSS_ENTROPY_DLL = get_module_path(CUDA_CROSS_ENTROPY_MODULE)

# Helper to load CUDA functions
def _load_cuda_function(module_path, function_name, arg_types):
    dll = ctypes.CDLL(module_path, mode=ctypes.RTLD_GLOBAL)
    func = getattr(dll, function_name)
    func.argtypes = arg_types
    return func

CUDA_CROSS_ENTROPY_FORWARD_BACKWARD = _load_cuda_function(
    CUDA_CROSS_ENTROPY_DLL, "cudaCrossEntropyForwardBackward", 
    [
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        c_int, 
        c_int,
        c_int,
        c_int,
        c_char,
        c_int,
    ]
)

def call_cuda_function(func, *args):
    # Helper for casting data to pointers
    def _to_pointer(obj: cp.ndarray):
        if obj is None:
            return None
        elif isinstance(obj, cp.ndarray):
            if isinstance(obj, bytes):
                return ctypes.c_char(obj)
            elif obj.dtype == cp.float32:
                return ctypes.cast(obj.data.ptr, POINTER(c_float))
            elif obj.dtype == cp.int32:
                return ctypes.cast(obj.data.ptr, POINTER(c_int))

        return obj

    return func(*[_to_pointer(arg) for arg in args])


def cross_entropy_forward_backward(
    logits: cp.ndarray, 
    labels: cp.ndarray, 
    reduction: str = "none", 
    ignore_index: int = -100
) -> cp.ndarray:
    """
    Computes the cross-entropy loss between logits and labels using CUDA.
    
    Args:
        logits (cp.ndarray): Logits tensor of shape (batch_size * seq_length, vocab_size).
        labels (cp.ndarray): Labels tensor of shape (batch_size * seq_length,).
        reduction (str): Reduction method. Can be 'none', 'mean', or 'sum'.
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
    """

    if logits.ndim != 2:
        raise ValueError("Logits must be 2D tensor")
    if labels.ndim != 1:
        raise ValueError("Labels must be 1D tensor")
    if labels.shape[0] != logits.shape[0]:
        raise ValueError("Logits and labels must have the same number of samples")
    if labels.dtype != "int32":
        raise TypeError("Labels must be of int32 dtype")
    if logits.dtype != "float32":
        raise TypeError("Logits must be of float32 dtype")
    if reduction not in ["none", "mean", "sum"]:
        raise ValueError("Reduction must be 'none', 'mean', or 'sum'")

    batch_sequence_size, vocab_size = logits.shape

    loss = cp.empty(batch_sequence_size, dtype=logits.dtype)
    lse = cp.empty(batch_sequence_size, dtype=logits.dtype)

    logits = cp.ascontiguousarray(logits)
    labels = cp.ascontiguousarray(labels)

    n_non_ignore = cp.sum(labels != ignore_index).item()

    reduction_char = {
        "none": b'n',
        "mean": b'm',
        "sum": b's'
    }[reduction]

    call_cuda_function(
        CUDA_CROSS_ENTROPY_FORWARD_BACKWARD,
        logits,
        loss,
        lse,
        labels,
        logits.strides[0] // logits.itemsize,    # check this
        ignore_index,
        batch_sequence_size,
        vocab_size,
        reduction_char, 
        n_non_ignore,
    )

    grad_logits = logits

    # loss = cp.where(labels == ignore_index, 0.0, loss) # set loss to 0 where labels are ignore_index

    if reduction == "mean":
        loss = cp.sum(loss) / cp.sum(labels != ignore_index) # mean loss, ignoring the ignore_index
    elif reduction == "sum":
        loss = cp.sum(loss)
    
    return loss, grad_logits



class _CUDACrossEntropyTensor(Tensor):
    def __init__(self, data, args, op, device):
        super().__init__(data, args, op, device=device)

        def grad_fn(y_pred: Tensor, grad_y_pred, grad):
            y_pred.apply_grad(grad_y_pred * grad)
           
        self.grad_fn = grad_fn

    def __call__(self, logits: Tensor, labels: Tensor) -> Tensor:
        return self.grad_fn(logits, labels)

class CUDACrossEntropyLoss(nn.Module):
    def __init__(self, reduction="none", ignore_index=-100):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        if not isinstance(y_pred, Tensor) or not isinstance(y_true, Tensor):
            raise TypeError("Input values must be tensors")
        if y_pred.device != y_true.device != "cuda":
            raise ValueError("Tensors must be on the cuda device")
        
        if y_pred.dtype != "float32":
            raise TypeError("Predictions must be of float32 dtype")

        if y_true.dtype != "int32":
            raise TypeError("Target must be of int32 dtype")
        
        loss, grad_y_pred = cross_entropy_forward_backward(
            y_pred.data,
            y_true.data,
            reduction=self.reduction,
            ignore_index=self.ignore_index
        )

        return _CUDACrossEntropyTensor(loss, (y_pred, grad_y_pred,), "cross_entropy", device="cuda")




