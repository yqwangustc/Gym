# Normalizer provenance

The English ASR normalizer's upstream is
[`whisper_normalizer`](https://github.com/kurianbenoy/whisper_normalizer),
which exposes `whisper_normalizer.english.EnglishTextNormalizer` and implements
Whisper's English text-normalization rules.

These modules preserve a pinned normalizer snapshot captured from the baseline
evaluation scorer. Keeping this snapshot stable makes new verification results
comparable with saved baseline predictions. The captured source snapshot's
Apache-2.0 license is retained in `LICENSE`. `PROVENANCE.json` records precise
origin identifiers, source paths, source and vendor SHA-256 hashes, the original
`EnglishTextNormalizer` line range, and replay evidence.

`asr_normalizer.py` and `english_abbreviations.py` are the complete captured
modules. `english_text_normalizer.py` contains the original normalizer class
extracted from `utils.py`, with imports retargeted to sibling modules. SPDX
notices and formatting are the only other edits. The abstract syntax trees of
both full modules and the extracted class were checked against the captured
originals and are identical. Unrelated video, subtitle, and evaluation
dependencies are omitted; the retained normalizer uses the Python standard
library.

The upstream project identifies the normalizer's origin; the captured files and
replay evidence pin this scoring profile's exact behavior. A newer upstream
release is not automatically equivalent to this snapshot. Keep these files
fixed for the default versioned profile. Changes to normalization require a
separate scoring version and a new parity baseline; existing scoring-version
identifiers remain unchanged for result compatibility.
