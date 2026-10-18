"""GLUE metric conventions and optimizer portability."""
import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
from experiments.train_glue import glue_metrics, torch_weight_decay
from experiments.nlp import read_json
from pathlib import Path


def test_metrics_and_paper_recipe():
    scores = np.array([[2., 0.], [0., 2.], [2., 0.], [0., 2.]])
    labels = np.array([0, 1, 0, 1])
    assert glue_metrics("mrpc", scores, labels) == {"accuracy": 1., "f1": 1.}
    assert glue_metrics("cola", scores, labels)["matthews_correlation"] == 1.
    corr = glue_metrics("stsb", np.array([[1.], [3.], [5.]]), np.array([0., 2., 4.]))
    assert corr["pearson"] == pytest.approx(1.)
    assert corr["spearmanr"] == pytest.approx(1.)
    recipe = read_json(Path(__file__).resolve().parents[2] / "experiments/configs/memsolve_bert_glue.json")
    assert {task for task, r in recipe["tasks"].items() if r["init"] == "mnli"} == {"mrpc", "rte", "stsb"}
    assert len(recipe["tasks"]) == 8


def test_composer_decay_conversion_at_multiple_learning_rates():
    task = {"optimizer": "composer_adamw", "lr": 5e-5, "weight_decay": 5e-6}
    for fraction in (0., .06, 1., .2):
        parameter = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=task["lr"] * fraction, weight_decay=torch_weight_decay(task))
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        torch.testing.assert_close(parameter.detach(), torch.tensor([2 * (1 - fraction * task["weight_decay"])], dtype=torch.float64))
    assert torch_weight_decay({**task, "optimizer": "adamw"}) == task["weight_decay"]
