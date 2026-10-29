# Vendored: Gorilla BFCL v3 AST checker

`ast_checker.py` is a vendored copy of the Berkeley Function Calling
Leaderboard v3 AST checker, kept verbatim so BFCL v3 scores stay comparable
with the reference evaluation.

- **Upstream project:** Gorilla / Berkeley Function Calling Leaderboard —
  https://github.com/ShishirPatil/gorilla
  (`berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py`)
- **License:** Apache-2.0 (upstream and this copy).
- **Copied from:** VLMEvalKitMcore `vlmeval/dataset/utils/bfcl_ast_checker.py`
  at commit `df7a6995`, SHA-256
  `5571b88eddc0e05faad185062366abaf65de57b0569aa5b86fc2a6aee0e2fb94`. That copy
  is the one the reference BFCLv3 numbers were scored with. Its docstring lists
  how it differs from upstream: Python-only type handling, and no
  `MODEL_CONFIG_MAPPING` or `Language` dependencies.

The file is that copy with a two-line Apache-2.0 attribution header prepended
(for the repository's copyright check; added 2026-10-29) and no other change:
SHA-256 with the header `0f1aecdc5fc88cea29ae490358f4fa0c6996bcfe96c7b149dd6104b0a44d66b9`; without the header (lines 3 onward) it
still hashes to the value above. It is excluded from ruff in `pyproject.toml` so formatting hooks leave
it unchanged. Do not edit it; change scoring in `../scoring.py` instead.
