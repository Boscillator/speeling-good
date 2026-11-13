import math

import pytest
import torch
from torch.nn import functional as F

from speeling_good.module import passage_weighted_cross_entropy


@pytest.mark.parametrize("weight", [0.0, 0.2, 1.0])
def test_mixed_passages_weight_tokens_and_gradients_without_reweighting_denominator(
    weight,
):
    # Target probabilities are 1/2, 1/4 in the clean passage and 1/3 in
    # the erroneous passage. Padding/special-token logits must not contribute.
    logits = torch.tensor(
        [
            [[math.log(2), 0, 0], [0, math.log(2), 0], [10, -10, 5]],
            [[0, 0, 0], [-10, 10, 5], [5, 10, -10]],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[0, 0, -100], [1, -100, -100]])
    loss = passage_weighted_cross_entropy(
        logits, labels, torch.tensor([True, False]), weight
    )

    assert loss.item() == pytest.approx(
        (weight * (math.log(2) + math.log(4)) + math.log(3)) / 3
    )
    loss.backward()
    expected_gradient = logits.detach().softmax(-1)
    expected_gradient[0, 0, 0] -= 1
    expected_gradient[0, 1, 0] -= 1
    expected_gradient[1, 0, 1] -= 1
    expected_gradient[labels == -100] = 0
    expected_gradient[0] *= weight
    torch.testing.assert_close(logits.grad, expected_gradient / 3)
    if weight == 1:
        torch.testing.assert_close(
            loss, F.cross_entropy(logits.transpose(1, 2), labels)
        )


@pytest.mark.parametrize("weight", [0.0, 0.2, 1.0])
def test_all_clean_batch_scales_loss_and_retains_gradient(weight):
    logits = torch.zeros((2, 3, 3), requires_grad=True)
    labels = torch.tensor([[0, 0, -100], [0, -100, -100]])
    loss = passage_weighted_cross_entropy(
        logits, labels, torch.tensor([True, True]), weight
    )

    assert loss.item() == pytest.approx(weight * math.log(3))
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    if weight == 0:
        assert torch.count_nonzero(logits.grad) == 0
    else:
        assert torch.count_nonzero(logits.grad) > 0
