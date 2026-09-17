"""CPU regression tests for repaired branches, without loading training backends."""

from collections.abc import Callable
import importlib.util
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


class AttentionArgs(TypedDict):
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    embed_dim_to_check: int
    num_heads: int
    in_proj_weight: torch.Tensor
    in_proj_bias: torch.Tensor
    bias_k: None
    bias_v: None
    add_zero_attn: bool
    dropout_p: float
    out_proj_weight: torch.Tensor
    out_proj_bias: torch.Tensor
    training: bool
    need_weights: bool


def load_source(relative_path):
    spec = importlib.util.spec_from_file_location(
        "regression_subject", ROOT / relative_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("add_zero_attn", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_attention_matches_torch_with_dropout_and_padding(add_zero_attn, training):
    functional = load_source("slowfast/models/torch_utils/functional.py")
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(4, 2, 8, generator=generator)
    kwargs = AttentionArgs(
        query=x,
        key=x,
        value=x,
        embed_dim_to_check=8,
        num_heads=2,
        in_proj_weight=torch.randn(24, 8, generator=generator),
        in_proj_bias=torch.randn(24, generator=generator),
        bias_k=None,
        bias_v=None,
        add_zero_attn=add_zero_attn,
        dropout_p=0.2,
        out_proj_weight=torch.randn(8, 8, generator=generator),
        out_proj_bias=torch.randn(8, generator=generator),
        training=training,
        need_weights=True,
    )
    with torch.random.fork_rng():
        torch.manual_seed(99)
        actual = functional.multi_head_attention_forward(**kwargs)
        torch.manual_seed(99)
        expected = torch.nn.functional.multi_head_attention_forward(**kwargs)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_five_position_crop_preserves_label_and_existing_offsets():
    transforms = load_source("slowfast/datasets/transforms.py")
    pixels = np.arange(8 * 16 * 3, dtype=np.uint8).reshape(8, 16, 3)
    frame = Image.fromarray(pixels)
    crops, label = transforms.GroupFCSample(8)(([frame], 7))
    offsets = transforms.GroupMultiScaleCrop.fill_fix_offset(False, 16, 8, 8, 8)
    assert len(crops) == len(offsets) == 5
    assert label == 7
    for crop, (x, y) in zip(crops, offsets):
        np.testing.assert_array_equal(np.asarray(crop), pixels[y : y + 8, x : x + 8])


def test_restored_activation_functions_match_torch():
    functional = load_source("slowfast/models/torch_utils/functional.py")
    x = torch.tensor([-2.0, 0.0, 3.0])
    for name in ("relu", "gelu", "silu", "mish", "softplus", "logsigmoid"):
        actual = getattr(functional, name)(x)
        expected = getattr(torch.nn.functional, name)(x)
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("clip_value", [0.0, 0.1])
def test_normal_training_branch_matches_sgd_step(monkeypatch, clip_value):
    """Execute the complete epoch function with CPU backend and meter adapters.

    Loading its AST avoids importing unrelated optional GPU training backends.
    The function body is compiled unchanged from the production source.
    """
    import ast
    from types import SimpleNamespace
    from unittest.mock import Mock

    monkeypatch.syspath_prepend(str(ROOT))
    from slowfast.config.defaults import get_cfg

    cfg = get_cfg()
    cfg.NUM_GPUS = 0
    cfg.MODEL.MODEL_NAME = "RegressionLinear"
    cfg.MODEL.FROZEN_BN = False
    cfg.MODEL.KEEP_RAW_MODEL = False
    cfg.MODEL.RECORD_ROUTING = False
    cfg.TRAIN.BATCH_SIZE = 2
    cfg.TRAIN.LINEAR_CONNECT_CLIMB = False
    cfg.TRAIN.ZERO_SHOT_META_LEARN = False
    cfg.TRAIN.CROSS_BATCH_META_LEARN = False
    cfg.TRAIN.MIXED_PRECISION = False
    cfg.TRAIN.EWC_SET = False
    cfg.MIXUP.ENABLE = False
    cfg.SOLVER.CLIP_GRAD_VAL = clip_value
    cfg.SOLVER.CLIP_GRAD_L2NORM = 0.0

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 6)

        def forward(self, inputs):
            return self.linear(inputs[0])

    with torch.random.fork_rng():
        torch.manual_seed(123)
        model = Model()
        reference = Model()
        reference.load_state_dict(model.state_dict())
        inputs = [torch.randn(2, 4)]
    labels = torch.tensor([1, 4])
    lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=lr)
    reference_loss = torch.nn.functional.cross_entropy(reference(inputs), labels)
    reference_loss.backward()
    if clip_value:
        torch.nn.utils.clip_grad_value_(reference.parameters(), clip_value)
    reference_optimizer.step()

    def grad_norm(parameters):
        return torch.stack(
            [p.grad.norm() for p in parameters if p.grad is not None]
        ).norm()

    namespace = dict(
        torch=torch,
        optim=SimpleNamespace(
            get_epoch_lr=lambda *args: lr,
            set_lr=lambda *args: None,
            get_grad_norm_=grad_norm,
        ),
        losses=SimpleNamespace(get_loss_func=lambda name: torch.nn.CrossEntropyLoss),
        misc=SimpleNamespace(check_nan_losses=lambda loss: None),
        metrics=load_source("slowfast/utils/metrics.py"),
        contrastive_parameter_surgery=lambda model, *args: (model, True),
    )
    source = ROOT / "tools/train_metazs.py"
    tree = ast.parse(source.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "train_epoch"
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    meter = Mock()
    # Disabled scaling follows the production full-precision branch.
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    cast(Callable[..., None], namespace["train_epoch"])(
        [(inputs, labels, torch.arange(2), torch.zeros(2), {})],
        model,
        optimizer,
        None,
        scaler,
        meter,
        0,
        cfg,
    )
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)
    assert meter.update_stats.call_args.args[5] == 2
