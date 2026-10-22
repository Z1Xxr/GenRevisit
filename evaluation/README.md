# Evaluation

`scripts/evaluate.py` dispatches benchmark-specific scoring of submitted images.
The adapter directories are `geneval2/`, `geneval/` and `genevalpp/`; GenEval2 is
implemented, while the other two directories reserve the same entry point.

## GenEval2 setup

Use the [official GenEval2](https://github.com/facebookresearch/GenEval2)
evaluator environment and install its dependencies following its README.
The verified scorer revision uses Transformers 4.57.0 and
`Qwen/Qwen3-VL-8B-Instruct`. Supply a local copy of that base model with
`--model-path` (the evaluator model, not the GenRevisit policy).

```bash
git clone https://github.com/facebookresearch/GenEval2.git external/GenEval2
```

The adapter reads `evaluation.py` from commit
`a6e82d2289e8d418f27f0adee77908b07060eea3` using `git show` and verifies its
SHA-256 before execution. It does not modify that checkout. The upstream scorer
is licensed under **CC BY-NC 4.0** and is not vendored here.

## Inputs

- `--benchmark-file`: benchmark JSONL with the original `prompt` and full
  `vqa_list` question/answer pairs for each row.
- `--manifest`: JSON array containing one record per benchmark item, as below.
  Row indices are zero-based; prompts must match exactly. Image paths resolve
  relative to the manifest. Replace the illustrative hash with the image's
  actual SHA-256 (for example, using `sha256sum`).

```json
[
  {
    "benchmark_row_index": 0,
    "original_prompt": "<exact benchmark prompt>",
    "image_path": "images/000000.png",
    "image_sha256": "<64-character SHA-256>",
    "outcome_kind": "submitted"
  }
]
```

Use `policy_invalid` or `no_submit` for non-submissions, with both image fields
set to `null`. Infrastructure failures must be retried, not recorded as policy
failures. Each submitted record refers to the agent's selected final image.

## Scoring and reports

Run the command in the root [README](../README.md#evaluation) from the repository
root. `prepare` accepts the same arguments as `score` and validates inputs without
loading the evaluator. Repeat `score` with identical inputs to resume. Keep each
output directory inside the repository, and use a new directory for changed
inputs or evaluator settings.

Results are written to `results/`, with aggregate reports under `reports/`.
The reported GenEval2 score is 100 times the mean of the per-image geometric
means over the full VQA probabilities. The report separately identifies
submission coverage, completion and the local full-cohort metric that assigns
zero to policy failures. Incomplete runs are not presented as complete scores.
