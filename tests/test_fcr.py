import torch

from networks.trainer import bernoulli_kl_from_logits, make_label_preserving_view
from validate import metrics_from_predictions


def test_bernoulli_kl_is_zero_for_identical_logits():
    logits = torch.tensor([-3.0, -0.2, 0.0, 1.4, 4.0], requires_grad=True)
    loss = bernoulli_kl_from_logits(logits, logits)
    assert torch.isfinite(loss)
    assert abs(loss.item()) < 1e-6


def test_bernoulli_kl_is_nonnegative_and_differentiable():
    first = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    second = torch.tensor([0.0, 1.5, -1.0], requires_grad=True)
    loss = bernoulli_kl_from_logits(first, second)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    loss.backward()
    assert first.grad is not None
    assert second.grad is not None


def test_label_preserving_view_keeps_batch_shape_and_finite_values():
    images = torch.randn(3, 3, 64, 64)
    transformed = make_label_preserving_view(images)
    assert transformed.shape == images.shape
    assert torch.isfinite(transformed).all()


def test_metrics_report_known_binary_predictions():
    metrics = metrics_from_predictions([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert metrics["accuracy"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
