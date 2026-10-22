#!/usr/bin/env python3
"""Isolated official submitted-image scoring; no evaluator imports before score.

prepare/score --manifest rows.json --benchmark-file official.jsonl --scorer-repo external/GenEval2 --model-path LOCAL
              --output-root REPO_PATH
report --output-root REPO_PATH

Manifest: a JSON array (or {"rows": [...]}) with zero-based benchmark_row_index,
original_prompt, image_path, image_sha256, outcome_kind. Outcomes are submitted,
policy_invalid, no_submit. Non-submissions require null image fields. Relative
image paths resolve against the input manifest, never against the scorer cwd.

One output root is one immutable evaluation: prepare can append disjoint rows
or repeat identical rows. A process lock and atomic artifacts ensure exactly-once
result publication. A crash before publication may repeat inference on resume;
no claim of exactly-once GPU execution is made. No best-image selection occurs.

Upstream GenEval2 scorer: CC BY-NC 4.0; see evaluation/README.md. The committed source is snapshotted verbatim at runtime, not copied
into production modules. Runtime incompatibilities fail without adaptations.
"""
from __future__ import annotations

import argparse
import hashlib
from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from unittest.mock import patch

from pathlib import PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run_relative_uri(uri: str) -> None:
    if not uri:
        raise ValueError("artifact uri must not be empty")
    if uri.startswith("/") or "://" in uri or "\\" in uri or ":" in uri:
        raise ValueError(f"artifact uri must be run-relative POSIX path: {uri}")
    path = PurePosixPath(uri)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"artifact uri must be run-relative POSIX path: {uri}")


def artifact_path(root: Path, uri: str) -> Path:
    validate_run_relative_uri(uri)
    return root.joinpath(*PurePosixPath(uri).parts)


def write_artifact_bytes(root: Path, uri: str, data: bytes) -> str:
    path = artifact_path(root, uri)
    expected_sha256 = sha256_bytes(data)
    if path.exists():
        existing_sha256 = sha256_file(path)
        if existing_sha256 != expected_sha256:
            raise ValueError(f"artifact conflict for {uri}: existing content differs")
        return existing_sha256

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as fh:
        fh.write(data)
    tmp_path.replace(path)
    return expected_sha256



SOURCE_COMMIT = "a6e82d2289e8d418f27f0adee77908b07060eea3"
SOURCE_SHA256 = "90791e35321a1a2bf65366517dd60f39037f90435490eb122c0027b52d42cda9"
OFFICIAL_BENCHMARK_SHA256 = "09233adbc9f877fba205f1a20efa4a8640e7e62a91dfd55893562bed8a928e43"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
OUTCOMES = {"submitted", "policy_invalid", "no_submit"}
ROW_FIELDS = {"benchmark_row_index", "original_prompt", "image_path", "image_sha256", "outcome_kind"}


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode()


def digest(value):
    return sha256_bytes(json_bytes(value))


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(data):
    def invalid(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=invalid)


def read_json(path):
    return parse_json(path.read_bytes())


def _inside(path, root):
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"output escapes repository/root: {path}")
    return path


@contextmanager
def output_lock(output_root):
    root = _inside(Path(output_root).absolute(), REPO_ROOT)
    if root.resolve() == REPO_ROOT:
        raise ValueError("output-root must be a dedicated repository subdirectory")
    root.mkdir(parents=True, exist_ok=True)
    lock_path = _inside(root / ".scoring.lock", root)
    with lock_path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield root
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def publish(root, uri, data):
    """Use the repository immutable helper under the shared process lock."""
    path = _inside(root / uri, root)
    _inside(path.with_name(f".{path.name}.tmp"), root)
    write_artifact_bytes(root, uri, data)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_json(root, uri, value):
    publish(root, uri, json_bytes(value))


def committed_source(scorer_repo):
    source_root = Path(scorer_repo).resolve(strict=True)
    result = subprocess.run(
        ["git", "-c", f"safe.directory={source_root}", "-C", str(source_root),
         "show", f"{SOURCE_COMMIT}:evaluation.py"],
        check=True, capture_output=True,
    )
    if sha256_bytes(result.stdout) != SOURCE_SHA256:
        raise ValueError("committed scorer source SHA256 mismatch")
    return result.stdout, str(source_root)


def benchmark_rows(data):
    rows = [parse_json(line) for line in data.splitlines() if line.strip()]
    prompts = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
            raise ValueError("benchmark row requires exact prompt")
        if row["prompt"] in prompts:
            raise ValueError("duplicate benchmark prompt")
        prompts.add(row["prompt"])
        vqas = row.get("vqa_list")
        if not isinstance(vqas, list) or not vqas or any(
            not isinstance(vqa, list) or len(vqa) != 2
            or any(not isinstance(part, str) for part in vqa) for vqa in vqas
        ):
            raise ValueError("benchmark requires nonempty VQA question/answer pairs")
    if not rows:
        raise ValueError("empty benchmark")
    if sha256_bytes(data) == OFFICIAL_BENCHMARK_SHA256:
        lengths = [len(row["vqa_list"]) for row in rows]
        if (len(rows), sum(lengths), min(lengths), max(lengths)) != (800, 6012, 3, 11):
            raise ValueError("official benchmark statistics mismatch")
    return rows


def verify_image(row):
    if row["outcome_kind"] != "submitted":
        return
    path = Path(row["image_path"])
    if not path.is_file() or sha256_file(path) != row["image_sha256"]:
        raise ValueError(f"infrastructure image missing/hash mismatch: row {row['benchmark_row_index']}")
    from PIL import Image
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    except Exception as exc:
        raise ValueError(f"infrastructure corrupt image: row {row['benchmark_row_index']}") from exc


def validate_row(row, benchmark, base):
    if not isinstance(row, dict) or set(row) != ROW_FIELDS:
        raise ValueError("manifest row must contain exactly the five documented fields")
    row = dict(row)
    index = row["benchmark_row_index"]
    if type(index) is not int or not 0 <= index < len(benchmark):
        raise ValueError("invalid zero-based benchmark_row_index")
    if row["original_prompt"] != benchmark[index]["prompt"]:
        raise ValueError(f"exact prompt/index mismatch: row {index}")
    if not isinstance(row["outcome_kind"], str) or row["outcome_kind"] not in OUTCOMES:
        raise ValueError("unknown outcome_kind (infrastructure failures are not policy outcomes)")
    if row["outcome_kind"] == "submitted":
        if not isinstance(row["image_path"], str) or not row["image_path"]:
            raise ValueError("submitted row requires image_path")
        if not isinstance(row["image_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["image_sha256"]):
            raise ValueError("submitted row requires lowercase image_sha256")
        row["image_path"] = str((base / row["image_path"]).resolve())
    elif row["image_path"] is not None or row["image_sha256"] is not None:
        raise ValueError("non-submission image fields must be null; no best-image substitution")
    verify_image(row)
    return row


def model_fingerprint(model_path):
    """Verify local base layout and hash all loader inputs, including weights.

    This is content pinning and architecture validation, not a claim that a
    local checkpoint was independently authenticated against Hugging Face.
    """
    path = Path(model_path).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("model-path must be an explicit local base model directory")
    config = read_json(path / "config.json")
    if config.get("model_type") != "qwen3_vl" or config.get("architectures") != ["Qwen3VLForConditionalGeneration"]:
        raise ValueError("model-path is not a Qwen3-VL base model")
    if list(path.glob("adapter*")):
        raise ValueError("model-path must be a base scorer, not an adapter")
    files = sorted(p for p in path.rglob("*") if p.is_file()
                   and not any(part.startswith(".") for part in p.relative_to(path).parts)
                   and p.suffix in {".json", ".safetensors", ".bin", ".txt", ".model", ".jinja", ".tiktoken"})
    weights = [p for p in files if p.suffix in {".safetensors", ".bin"}]
    if not weights or any(p.stat().st_size == 0 for p in weights):
        raise ValueError("local model weights missing or empty")
    for index in path.glob("*.index.json"):
        mapping = read_json(index).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("invalid model shard index")
        for shard in mapping.values():
            if not isinstance(shard, str) or Path(shard).name != shard or path / shard not in weights:
                raise ValueError("model shard index references missing/unsafe weights")
    if not (path / "tokenizer_config.json").is_file() or not any(
        (path / name).is_file() for name in ("preprocessor_config.json", "processor_config.json")
    ):
        raise ValueError("local processor/tokenizer configuration missing")
    catalog = [{"path": str(p.relative_to(path)), "size": p.stat().st_size,
                "sha256": sha256_file(p)} for p in files]
    return {"path": str(path), "files": catalog, "content_sha256": digest(catalog),
            "verification": "local base architecture/layout and full loader-file hashes; no remote authentication"}


def runtime_fingerprint():
    packages = {}
    for name in ("torch", "torchvision", "transformers", "numpy", "scipy", "Pillow",
                 "accelerate", "tokenizers", "safetensors", "huggingface-hub", "tqdm", "qwen-vl-utils"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "upstream_transformers_pin": "4.57.0",
            "transformers_matches_upstream_pin": packages["transformers"] == "4.57.0",
            "runtime_adaptations": [],
            "environment": {key: os.environ.get(key) for key in (
                "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE",
                "CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF")}}


def loader_resolution(model_path):
    return {"only_original_id": MODEL_ID, "resolved_local_path": str(model_path),
            "added_kwargs": {"local_files_only": True},
            "preserved_processor_kwargs": {"torch_dtype": "auto", "device_map": "auto"},
            "preserved_model_kwargs": {"dtype": "auto", "device_map": "auto"},
            "scoring_logic_modified": False}


def _prepare(root, manifest, benchmark_path, model_path, scorer_repo):
    manifest = Path(manifest).resolve(strict=True)
    raw_manifest = manifest.read_bytes()
    value = parse_json(raw_manifest)
    if isinstance(value, dict) and set(value) == {"rows"}:
        value = value["rows"]
    if not isinstance(value, list) or not value:
        raise ValueError("manifest requires a nonempty JSON row array")
    benchmark_data = Path(benchmark_path).read_bytes()
    benchmark = benchmark_rows(benchmark_data)
    rows = [validate_row(row, benchmark, manifest.parent) for row in value]
    if len({row["benchmark_row_index"] for row in rows}) != len(rows):
        raise ValueError("duplicate manifest benchmark_row_index")
    source, source_root = committed_source(scorer_repo)
    model = model_fingerprint(model_path)
    freeze = {"format": "genrevisit_geneval2@1", "source_commit": SOURCE_COMMIT,
              "source_sha256": SOURCE_SHA256, "source_root": source_root,
              "license": "CC BY-NC 4.0", "benchmark_sha256": sha256_bytes(benchmark_data),
              "official800": sha256_bytes(benchmark_data) == OFFICIAL_BENCHMARK_SHA256,
              "benchmark_rows": len(benchmark), "benchmark_vqas": sum(len(b["vqa_list"]) for b in benchmark),
              "model": model, "runtime": runtime_fingerprint(),
              "wrapper_sha256": sha256_file(Path(__file__)),
              "loader_resolution": loader_resolution(model["path"])}
    if (root / "freeze.json").exists() and read_json(root / "freeze.json") != freeze:
        raise ValueError("frozen scorer/runtime/model/benchmark mismatch; use a new output root")
    for row in rows:
        path = root / f"requests/{row['benchmark_row_index']:06d}.json"
        if path.exists() and read_json(path) != row:
            raise ValueError(f"immutable input conflict: row {row['benchmark_row_index']}")
    publish(root, "scorer/evaluation.py", source)
    publish(root, "benchmark.jsonl", benchmark_data)
    publish_json(root, "freeze.json", freeze)
    publish(root, f"manifests/{sha256_bytes(raw_manifest)}.json", raw_manifest)
    for row in rows:
        publish_json(root, f"requests/{row['benchmark_row_index']:06d}.json", row)
    return freeze, benchmark, rows


def prepare(manifest, benchmark, output_root, model_path, scorer_repo):
    with output_lock(output_root) as root:
        freeze, _, rows = _prepare(root, manifest, benchmark, model_path, scorer_repo)
        return {"prepared_rows": len(rows), "freeze_sha256": digest(freeze),
                "official800": freeze["official800"], "models_loaded": False}


def load_scorer(source, model_path):
    """Execute unchanged, verified source with only the two exact-ID loaders redirected."""
    if sha256_bytes(source) != SOURCE_SHA256:
        raise ValueError("scorer source SHA256 mismatch")
    import torch
    import numpy as np
    import transformers
    # Upstream processor bridges NumPy and Torch. Detect broken ABI; never shim it.
    try:
        torch.from_numpy(np.zeros(1, dtype=np.float32)).numpy()
    except Exception as exc:
        raise RuntimeError("Torch/NumPy interoperability failed; no scoring adaptation applied") from exc
    execution = {"torch_version": str(torch.__version__), "cuda": torch.version.cuda,
                 "hip": torch.version.hip,
                 "devices": [str(torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())],
                 "default_dtype": str(torch.get_default_dtype()),
                 "matmul_precision": torch.get_float32_matmul_precision(),
                 "runtime_adaptations": []}
    calls = []

    def redirect(original, kind):
        def call(identifier, *args, **kwargs):
            if identifier != MODEL_ID:
                raise ValueError("refusing loader override for non-official model ID")
            calls.append({"loader": kind, "original_id": identifier,
                          "local_path": str(model_path), "original_kwargs": dict(kwargs),
                          "added_kwargs": {"local_files_only": True}})
            return original(str(model_path), *args, **kwargs, local_files_only=True)
        return call

    processor = transformers.AutoProcessor
    model = transformers.Qwen3VLForConditionalGeneration
    namespace = {"__name__": "_verified_geneval2_scorer", "__file__": "scorer/evaluation.py"}
    with patch.object(processor, "from_pretrained", redirect(processor.from_pretrained, "processor")), \
         patch.object(model, "from_pretrained", redirect(model.from_pretrained, "model")):
        exec(compile(source, "scorer/evaluation.py", "exec"), namespace)
    execution["loader_calls"] = calls
    if [call["loader"] for call in calls] != ["processor", "model"]:
        raise ValueError("unexpected official loader calls")
    return namespace["soft_tifa"], execution


def score_values(am, probabilities, count):
    from scipy.stats import gmean
    if not isinstance(probabilities, list) or len(probabilities) != count:
        raise ValueError("scorer probability count does not match full VQA list")
    # Upstream sums answer-token entries without deduplication, so values above
    # one are preserved rather than clipped. Finite nonnegative values required.
    if any(type(p) not in (float, int) or not math.isfinite(p) or p < 0 for p in probabilities):
        raise ValueError("invalid scorer probabilities")
    expected_am = sum(probabilities) / count
    if type(am) not in (float, int) or not math.isfinite(am) or not math.isclose(am, expected_am, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("scorer arithmetic mean does not match probabilities")
    gm = float(gmean(probabilities))
    if not math.isfinite(gm):
        raise ValueError("nonfinite official GM")
    return {"soft_tifa_am": am, "probabilities": probabilities, "official_gm": gm}


def result_record(freeze, row, benchmark_row, values, execution_sha256):
    result = {"freeze_sha256": digest(freeze), "request_sha256": digest(row),
            "benchmark_row_index": row["benchmark_row_index"], "original_prompt": row["original_prompt"],
            "image_path": row["image_path"], "image_sha256": row["image_sha256"],
            "outcome_kind": row["outcome_kind"], "vqa_count": len(benchmark_row["vqa_list"]),
            "vqa_sha256": digest(benchmark_row["vqa_list"]), "execution_sha256": execution_sha256,
            **values}
    return {**result, "result_sha256": digest(result)}


NULL_VALUES = {"soft_tifa_am": None, "probabilities": None, "official_gm": None}


def validate_result(result, freeze, row, benchmark_row, execution_sha256):
    if row["outcome_kind"] == "submitted":
        if execution_sha256 is None:
            raise ValueError("scored result missing execution fingerprint")
        values = score_values(result.get("soft_tifa_am"), result.get("probabilities"), len(benchmark_row["vqa_list"]))
    else:
        values = NULL_VALUES
        execution_sha256 = None
    expected = result_record(freeze, row, benchmark_row, values, execution_sha256)
    if result != expected:
        raise ValueError(f"cached result mismatch: row {row['benchmark_row_index']}")


def score(manifest, benchmark, output_root, model_path, scorer_repo):
    with output_lock(output_root) as root:
        freeze, benchmarks, rows = _prepare(root, manifest, benchmark, model_path, scorer_repo)
        execution_path = root / "execution.json"
        execution_hash = digest(read_json(execution_path)) if execution_path.exists() else None
        scorer = None
        written = skipped = 0
        for row in sorted(rows, key=lambda r: r["benchmark_row_index"]):
            index = row["benchmark_row_index"]
            path = root / f"results/{index:06d}.json"
            if path.exists():
                validate_result(read_json(path), freeze, row, benchmarks[index], execution_hash)
                skipped += 1
                continue
            values = NULL_VALUES
            row_execution = None
            if row["outcome_kind"] == "submitted":
                if scorer is None:
                    scorer, execution = load_scorer((root / "scorer/evaluation.py").read_bytes(), freeze["model"]["path"])
                    publish_json(root, "execution.json", execution)
                    execution_hash = digest(execution)
                verify_image(row)
                am, probabilities = scorer(benchmarks[index]["vqa_list"], row["image_path"])
                verify_image(row)
                values = score_values(am, probabilities, len(benchmarks[index]["vqa_list"]))
                row_execution = execution_hash
            result = result_record(freeze, row, benchmarks[index], values, row_execution)
            publish_json(root, f"results/{index:06d}.json", result)
            written += 1
        return {"written": written, "validated_skipped": skipped, "models_loaded": scorer is not None}


def report(output_root):
    with output_lock(output_root) as root:
        freeze = read_json(root / "freeze.json")
        source = (root / "scorer/evaluation.py").read_bytes()
        data = (root / "benchmark.jsonl").read_bytes()
        if sha256_bytes(source) != SOURCE_SHA256 or freeze["source_sha256"] != SOURCE_SHA256:
            raise ValueError("frozen scorer hash mismatch")
        if sha256_bytes(data) != freeze["benchmark_sha256"]:
            raise ValueError("frozen benchmark hash mismatch")
        benchmark = benchmark_rows(data)
        execution_path = root / "execution.json"
        execution_hash = digest(read_json(execution_path)) if execution_path.exists() else None
        results = []
        requests = {}
        for path in sorted((root / "requests").glob("*.json")):
            row = validate_row(read_json(path), benchmark, root)
            index = row["benchmark_row_index"]
            if path.name != f"{index:06d}.json" or index in requests:
                raise ValueError("request file/index mismatch")
            requests[index] = row
        for path in sorted((root / "results").glob("*.json")):
            result = read_json(path)
            index = result["benchmark_row_index"]
            if type(index) is not int or index not in requests or path.name != f"{index:06d}.json":
                raise ValueError("orphan or misindexed result")
            validate_result(result, freeze, requests[index], benchmark[index], execution_hash)
            results.append(result)
        counts = {kind: sum(row["outcome_kind"] == kind for row in requests.values()) for kind in sorted(OUTCOMES)}
        valid = [result["official_gm"] for result in results if result["outcome_kind"] == "submitted"]
        total = len(benchmark)
        complete = len(results) == len(requests) == total
        summary = {"freeze_sha256": digest(freeze), "official800": freeze["official800"],
                   "benchmark_rows": total, "prepared_rows": len(requests), "completed_rows": len(results),
                   "pending_rows": len(requests) - len(results), "unprepared_rows": total - len(requests),
                   "outcome_counts": counts, "scored_submissions": len(valid),
                   "submission_coverage": counts["submitted"] / total,
                   "scored_submission_coverage": len(valid) / total,
                   "valid_submission_scoring_complete": len(valid) == counts["submitted"],
                   "full_cohort_complete": complete,
                   "official_conditional_valid_submission_gm_percent": 100 * sum(valid) / len(valid) if valid else None,
                   "local_policy_zero_penalized_full_cohort_gm_percent": 100 * sum(valid) / total if complete else None,
                   "local_policy_metric_definition": "official row GM; policy_invalid/no_submit=0; only defined for complete cohort",
                   "aggregation": "100 * mean(scipy.stats.gmean(full per-row probability list)); no epsilon floor",
                   "runtime": freeze["runtime"], "loader_resolution": freeze["loader_resolution"]}
        publish_json(root, f"reports/{digest(summary)}.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "score", "report"):
        command = commands.add_parser(name)
        command.add_argument("--output-root", type=Path, required=True)
        if name != "report":
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--benchmark-file", dest="benchmark", type=Path, required=True)
            command.add_argument("--scorer-repo", type=Path, required=True)
            command.add_argument("--model-path", type=Path, required=True,
                                 help="Caller-verified local base Qwen3-VL-8B-Instruct; content pinned without remote authentication")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        result = {"prepare": prepare, "score": score, "report": report}[command](**args)
    except Exception as exc:
        # Scorer errors may contain input tensors/paths; do not persist raw errors.
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
