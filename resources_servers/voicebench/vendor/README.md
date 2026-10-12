# VoiceBench rule scorer provenance

These files are derived from a captured VoiceBench reference-evaluator
implementation. The reference evaluator is the pre-existing implementation
used to establish the baseline parsing and scoring behavior. Its exact source
commit, original file paths, SHA-256 hashes, and mapping to vendored files are
recorded in [PROVENANCE.json](PROVENANCE.json). The commit identifies that
captured implementation; it is not a claim that the files match the latest
public release. [LICENSE](LICENSE) contains the Apache-2.0 license, and the
Google Research copyright and license notices remain in the
instruction-following source files.

Only the rule parsers and instruction-check dependencies are included. Dataset
aggregate methods and their NumPy/Loguru imports are removed from these copies;
the resource server aggregates each scorer's sufficient statistics. BBH's
fallback console prints are replaced by explicit scorer diagnostics. Files are
formatted, and random draws used by BBH and randomized instruction defaults are
routed through `randomness.py` to the supplied evaluation-wide `random.Random`.

The `legacy_v1` value is a retained compatibility identifier for this fixed
baseline scoring profile. It preserves the following parsing and scoring rules:

- MCQ parsing follows the original ordered templates and guesses A–D on failure.
- BBH routes by the original task ID, guesses a binary answer on failure, and
  preserves the bare `b` Hyperbaton mapping.
- Refusal detection retains the original phrase list, duplicate/concatenated
  entries, and the rule that an empty answer is safe. Candidate health is checked
  separately by the resource server.
- IFEval preserves falsy-kwargs filtering, response-marker normalization, and
  strict/loose instruction checks. Dataset metrics require separate prompt and
  instruction denominators. Language-detection exceptions count as following,
  and `langdetect` retains its original per-detector stochastic behavior.

IFEval needs `absl-py`, `immutabledict`, `langdetect`, and `nltk`, plus local NLTK
Punkt resources. The scorer never downloads them during a verification request.
Other rule families do not import the IFEval dependency tree.

The rule tests include fixed expected-output regression cases and optional
independent checks against the captured, unmodified reference implementation.
If you have that source snapshot, set `VOICEBENCH_REFERENCE_ROOT` to its checkout
to enable the additional checks. Source-file hashes must match the provenance
record before the tests import the reference evaluators. The default regression
tests do not require a reference checkout.
