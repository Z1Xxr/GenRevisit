from __future__ import annotations

import importlib.util

import json

import multiprocessing

from pathlib import Path

import subprocess

import sys

import types

import pytest

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "evaluation/geneval2/scorer.py"

@pytest.fixture
def wrapper():
    spec = importlib.util.spec_from_file_location("geneval2_scoring", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.fixture
def case(tmp_path, monkeypatch, wrapper):
    monkeypatch.setattr(wrapper, "REPO_ROOT", tmp_path)
    source = b"mock scorer source"
    monkeypatch.setattr(wrapper, "SOURCE_SHA256", wrapper.sha256_bytes(source))
    monkeypatch.setattr(wrapper, "committed_source", lambda scorer_repo: (source, "/read-only/source"))
    monkeypatch.setattr(wrapper, "runtime_fingerprint", lambda: {"packages": {"transformers": "mock"}})
    model = tmp_path / "base"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"]}))
    (model / "tokenizer_config.json").write_text("{}")
    (model / "preprocessor_config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"mock model weights")
    benchmark = tmp_path / "benchmark.jsonl"
    benchmarks = [{"prompt": f"Prompt {i}", "atom_count": 1,
                   "vqa_list": [[f"Question {j}", "yes"] for j in range(i + 3)]} for i in range(3)]
    benchmark.write_text("".join(json.dumps(row) + "\n" for row in benchmarks))
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "red").save(image)
    rows = [{"benchmark_row_index": i, "original_prompt": row["prompt"], "image_path": "image.png",
             "image_sha256": wrapper.sha256_file(image), "outcome_kind": "submitted"}
            for i, row in enumerate(benchmarks)]
    manifest = tmp_path / "rows.json"
    manifest.write_text(json.dumps(rows))
    calls = []
    loads = []

    def scorer(vqas, path):
        calls.append((vqas, path))
        probabilities = [0.25] + [1.0] * (len(vqas) - 1)
        return sum(probabilities) / len(probabilities), probabilities

    def loader(source, path):
        loads.append(path)
        return scorer, {"mock_execution": True}

    monkeypatch.setattr(wrapper, "load_scorer", loader)
    return types.SimpleNamespace(
        args={"manifest": manifest, "benchmark": benchmark, "output_root": tmp_path / "scores", "model_path": model, "scorer_repo": tmp_path / "scorer"},
        rows=rows, calls=calls, loads=loads, model=model, image=image, root=tmp_path / "scores",
        manifest=manifest, benchmarks=benchmarks, source=source,
    )

def test_prepare_report_cpu_only_and_full_vqa_score_resume(case, wrapper):
    assert wrapper.prepare(**case.args)["models_loaded"] is False
    initial = wrapper.report(case.root)
    assert initial["pending_rows"] == 3
    assert initial["official_conditional_valid_submission_gm_percent"] is None
    assert initial["local_policy_zero_penalized_full_cohort_gm_percent"] is None
    assert not case.loads
    assert wrapper.score(**case.args)["written"] == 3
    assert [len(call[0]) for call in case.calls] == [3, 4, 5]
    assert all(Path(call[1]) == case.image for call in case.calls)
    assert wrapper.score(**case.args) == {"written": 0, "validated_skipped": 3, "models_loaded": False}
    assert len(case.loads) == 1
    report = wrapper.report(case.root)
    expected = 100 * sum(0.25 ** (1 / n) for n in (3, 4, 5)) / 3
    assert report["official_conditional_valid_submission_gm_percent"] == pytest.approx(expected)
    assert report["local_policy_zero_penalized_full_cohort_gm_percent"] == pytest.approx(expected)
    assert report["full_cohort_complete"] is True
    assert report["official800"] is False

def test_interrupted_scoring_preserves_published_rows(case, wrapper, monkeypatch):
    original_loader = wrapper.load_scorer

    def interrupted(source, model):
        scorer, execution = original_loader(source, model)

        def call(vqas, path):
            if len(vqas) == 4:
                raise RuntimeError("mock infrastructure interruption")
            return scorer(vqas, path)
        return call, execution

    monkeypatch.setattr(wrapper, "load_scorer", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        wrapper.score(**case.args)
    assert len(list((case.root / "results").glob("*.json"))) == 1
    assert wrapper.report(case.root)["local_policy_zero_penalized_full_cohort_gm_percent"] is None
    monkeypatch.setattr(wrapper, "load_scorer", original_loader)
    assert wrapper.score(**case.args)["validated_skipped"] == 1
    assert len(case.calls) == 3

def test_exact_zero_and_upstream_nonclipping(wrapper):
    assert wrapper.score_values(0.5, [0.0, 1.0], 2)["official_gm"] == 0
    assert wrapper.score_values(1.5, [1.0, 2.0], 2)["official_gm"] == pytest.approx(2 ** 0.5)
    for am, probabilities in [(0, [-1]), (0, [float("nan")]), (0.1, [0.5])]:
        with pytest.raises(ValueError):
            wrapper.score_values(am, probabilities, 1)

def test_source_hash_verified_before_import(wrapper):
    with pytest.raises(ValueError, match="source SHA256"):
        wrapper.load_scorer(b"untrusted source", "/never/load")
