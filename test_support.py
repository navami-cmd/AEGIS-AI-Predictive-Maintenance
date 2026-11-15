"""Lets the pure-Python tests import mc_utils even when PyTorch is not installed
(only a stub; real PyTorch is always used when it is available)."""
import sys
import types


def ensure_torch_importable():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        t = types.ModuleType("torch")
        nn = types.ModuleType("torch.nn")
        nn.Module = object
        nn.Dropout = type("Dropout", (), {})
        nn.LSTM = type("LSTM", (), {})
        t.nn = nn
        t.Tensor = type("Tensor", (), {})
        t.__version__ = "stub"
        sys.modules["torch"] = t
        sys.modules["torch.nn"] = nn
        return False
