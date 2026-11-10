import numpy as np
import torch
from models import build_model
from metrics import classification_metrics


def test_metrics_missing_class_and_confidence_one():
    result = classification_metrics(np.array([[1000., -1000., -1000.]]), [0], 3)
    assert result['accuracy'] == 1
    assert result['macro_f1'] == 1 / 3
    assert result['ece'] == 0


def test_identity_initializations():
    spec = {'methods': {'lora': {'rank': 8, 'alpha': 8, 'dropout': 0.0}, 'bottleneck_adapter': {'bottleneck_dim': 64}}}
    torch.manual_seed(42)
    base = build_model(spec, 'linear_probe', 3, smoke=True).eval()
    x = torch.randn(2, 3, 32, 32)
    for method in ['lora', 'bottleneck_adapter']:
        torch.manual_seed(42)
        net = build_model(spec, method, 3, smoke=True).eval()
        with torch.no_grad():
            torch.testing.assert_close(base(x), net(x))
        assert all(not p.requires_grad for n, p in net.named_parameters() if 'norm' in n)
