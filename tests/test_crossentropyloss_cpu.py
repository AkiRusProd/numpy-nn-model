import numpy as np
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neunet
import neunet.nn as nn
from neunet.autograd import Tensor


@pytest.mark.parametrize(
    "batch_size,classes,ignore_index,rtol,atol",
    [
        (16, 32, -100, 1e-5, 1e-5),
        (8, 64, -1, 1e-5, 1e-5),
    ],
)
def test_crossentropy_cpu_mean(batch_size, classes, ignore_index, rtol, atol):
    np.random.seed(42)
    logits = np.random.randn(batch_size, classes).astype(np.float32)
    labels = np.random.randint(0, classes, size=(batch_size,)).astype(np.int64)

    n_logits_ce = Tensor(logits.copy(), device="cpu", requires_grad=True)
    n_labels_ce = Tensor(labels.copy(), device="cpu", requires_grad=False, dtype=np.int64)

    n_logits_combo = Tensor(logits.copy(), device="cpu", requires_grad=True)
    n_labels_combo = Tensor(labels.copy(), device="cpu", requires_grad=False, dtype=np.int64)

    ce_loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_index)
    log_softmax = nn.LogSoftmax(axis=1)
    nll_loss = nn.NLLLoss(reduction="mean", ignore_index=ignore_index)

    loss_ce = ce_loss(n_logits_ce, n_labels_ce)
    loss_combo = nll_loss(log_softmax(n_logits_combo), n_labels_combo)

    assert np.allclose(loss_ce.data, loss_combo.data, rtol=rtol, atol=atol)

    loss_ce.backward()
    loss_combo.backward()

    assert np.allclose(n_logits_ce.grad, n_logits_combo.grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "batch_size,classes,ignore_index,rtol,atol",
    [
        (16, 32, -100, 1e-5, 1e-5),
        (8, 64, -1, 1e-5, 1e-5),
    ],
)
def test_crossentropy_cpu_none(batch_size, classes, ignore_index, rtol, atol):
    np.random.seed(123)
    logits = np.random.randn(batch_size, classes).astype(np.float32)
    labels = np.random.randint(0, classes, size=(batch_size,)).astype(np.int64)

    n_logits_ce = Tensor(logits.copy(), device="cpu", requires_grad=True)
    n_labels_ce = Tensor(labels.copy(), device="cpu", requires_grad=False, dtype=np.int64)

    n_logits_combo = Tensor(logits.copy(), device="cpu", requires_grad=True)
    n_labels_combo = Tensor(labels.copy(), device="cpu", requires_grad=False, dtype=np.int64)

    ce_loss = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
    log_softmax = nn.LogSoftmax(axis=1)
    nll_loss = nn.NLLLoss(reduction="none", ignore_index=ignore_index)

    loss_ce = ce_loss(n_logits_ce, n_labels_ce)
    loss_combo = nll_loss(log_softmax(n_logits_combo), n_labels_combo)

    assert np.allclose(loss_ce.data, loss_combo.data, rtol=rtol, atol=atol)

    loss_ce.backward()
    loss_combo.backward()

    assert np.allclose(n_logits_ce.grad, n_logits_combo.grad, rtol=rtol, atol=atol)
