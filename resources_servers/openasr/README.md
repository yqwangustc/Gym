# OpenASR

This stateless resource server evaluates English automatic speech recognition
(ASR) with word error rate (WER). It supports eight test splits: AMI,
Earnings22, GigaSpeech, LibriSpeech test-clean, LibriSpeech test-other,
SPGISpeech, TED-LIUM, and VoxPopuli. Each named resource instance evaluates one
split; multiple instances can run together with separate processes and ports.

The **reference** is the dataset's expected transcript, supplied in
`verifier_metadata.text`. The **hypothesis** is the transcript extracted from
the model's response. The verifier normalizes both transcripts, aligns their
words, and returns error counts. It preserves the raw response messages and
reasoning, the extracted answer, and both normalized transcripts for inspection.

Use an agent such as `simple_agent` to request a model response and send it to
this verifier. For archived audio, the audio-capable Gym vLLM adapter resolves
archive/member references before inference. This resource server reads text
labels and responses; it performs no dataset downloads, audio decoding, or
model calls.

## Configuration and data

Set `subset` to one of `ami`, `earnings22`, `gigaspeech`,
`librispeech_test_clean`, `librispeech_test_other`, `spgispeech`, `tedlium`, or
`voxpopuli`, and set a positive `expected_num_rows` for that split. The default
`scoring_version: asr_v1` selects the pinned English normalization,
transcript-extraction, unit-cost word alignment, and corpus WER profile.

Each Gym dataset row supplies `responses_create_params` for the agent and
`verifier_metadata` containing `subset`, a nonempty source `id`, and reference
`text`. Keep the reference label outside the model-visible input. A missing
reference is represented by `text: null` and is explicitly skipped during
scoring.

Optional `expected_sample_ids_file` identifies the exact samples expected in
the evaluation. It accepts a Gym JSONL, a JSON array of source IDs, or one
source ID per line. IDs must be unique and their count must equal
`expected_num_rows`; Gym JSONL subset identities must match the resource
instance. Without this file, coverage checks use the configured count rather
than an exact expected-ID set.

The five rows in `data/example.jsonl` ask a model to repeat synthetic phrases.
They are self-contained examples for exercising the request schema, agent to
verifier flow, normalization, and WER aggregation with a text-capable model.
Their phrases are deliberately visible in the input; they contain no audio
reference and do not test speech recognition or archive loading. The
`librispeech_test_clean` subset value supplies a supported configuration label;
these rows are not recordings from that benchmark.

`configs/openasr.yaml` connects those examples to `simple_agent` and a model
server named `policy_model`. A real audio evaluation needs a dataset whose
`responses_create_params.metadata.audio_ref` identifies an archive and member,
an audio-capable model adapter/backend, and the real split's expected count.
The adapter loads and encodes audio at inference time. The reference transcript
belongs only in `verifier_metadata.text`, outside the model-visible input.
Prepared datasets, audio archives, and runtime configs can live wherever your
deployment requires; this component does not require another repository or a
specific filesystem layout.

## Normalization and word errors

The English normalizer's upstream is
[whisper_normalizer](https://github.com/kurianbenoy/whisper_normalizer).
The resource vendors a pinned snapshot captured from the baseline evaluation
scorer so that cached predictions remain comparable. The snapshot handles
numbers, contractions, titles, spelling mappings, and bracketed noise. It uses
the Python standard library and adds no normalizer package dependency. See
`vendor/README.md` and `vendor/PROVENANCE.json` for attribution, hashes, and
validation evidence.

The final assistant answer supplies the hypothesis. If it contains closing
`</think>` tags, only text after the last closing tag is scored; separate
reasoning content remains available for inspection. A reasoning-only response
produces an empty hypothesis. Untagged final text is retained, including any
literal `audio is` prefix. The verifier does not strip such a prefix.

Alignment uses
[`rapidfuzz.distance.Levenshtein.editops`](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html#editops)
from pinned `rapidfuzz==3.14.6`, released under the
[MIT license](https://github.com/rapidfuzz/RapidFuzz/blob/main/LICENSE).
The inputs are `normalized_reference.split()` and
`normalized_hypothesis.split()`, so each token is a word. Each substitution,
deletion, or insertion costs one word error:

- **Substitution:** a reference word is replaced by another word.
- **Deletion:** a reference word is missing from the hypothesis.
- **Insertion:** the hypothesis contains an additional word.

The per-recording WER is the total errors divided by the reference-word count.
The authoritative dataset metrics are `openasr/corpus_wer` (fraction) and
`openasr/corpus_wer_percent`:

```text
corpus WER = sum(substitutions + deletions + insertions) / sum(reference words)
```

Corpus WER weights recordings by their reference-word counts; it is not the
mean of recording-level WERs. WER can exceed 100% when there are many
insertions. Per-row `reward` is a separate binary measure: it is one only for
exact normalized token equality with a completed candidate and a valid
reference.

Equal-cost alignments follow the pinned library's tie rule. S/D/I and
matching-word diagnostics can differ from the former Python substitution-first
alignment while total edits, reference-word counts, and WER remain identical.
Historical saved outputs retain their original diagnostics.

## Aggregation and completeness

The `/aggregate_metrics` hook sums the word counts and errors from received
verification records. It also reports source-ID coverage, duplicates,
missing/unexpected IDs, invalid verification rows, candidate completion, and
the effective denominator. The profile expects one rollout per source sample.

`openasr/complete` requires the configured unique sample coverage, no invalid
verification records, completed candidates, and a positive reference-word
denominator. With an expected-ID file, coverage also requires the exact sample
set. Masked verifier failures remain visible in coverage so an evaluation with
failed verification cannot appear complete. Report the completeness and
coverage fields alongside WER when assessing an evaluation result.

An incomplete candidate can still have a measured transcript and WER, but its
exact-match reward is zero and the evaluation is marked incomplete. A completed
empty answer deletes every reference word. References that normalize to zero
words still contribute hypothesis insertions to the corpus numerator; their
per-recording WER is undefined (`null`). A zero corpus denominator returns
`0.0` together with `openasr/wer_denominator_valid: false`, and the evaluation
is never marked complete.

## Dependencies and tests

Gym installs this component's `requirements.txt` before starting its server.
By default in an editable checkout, its interpreter is
`resources_servers/openasr/.venv/bin/python`. With
`uv_venv_dir: /path/to/venvs`, it is
`/path/to/venvs/resources_servers/openasr/.venv/bin/python`. All OpenASR
instances share that implementation environment; other resource
implementations have their own environments. The editable Gym requirement
supplies the framework dependencies. `editdistance==0.8.1` is an independent
distance oracle for tests and is not used in the serving path.

`skip_venv_if_present: true` checks for the interpreter and activation script
without refreshing installed dependencies. After changing requirements,
install them into the selected environment or prefetch with reuse disabled:

```bash
gym env prefetch --config path/to/graph.yaml --config path/to/runtime.yaml \
  skip_venv_if_present=false
```

Restart the resource processes to load changed code or imports. Editing the
manifest does not hot-reload a running server.

Run the focused suite through Gym's component setup:

```bash
gym env test --resources-server openasr
```

The 29 tests cover corpus weighting, unbounded insertion errors, empty and
missing labels/answers, refusal content, incomplete responses, masked
verification failures, sample coverage, final-answer extraction, Unicode and
whitespace tokenization, ambiguous alignments, and a 5,000-word transcript.

An independent replay of 2,620 saved baseline predictions reproduces every
normalized reference/hypothesis and per-row edit distance: 795 edits over
53,027 reference words, or `1.4992362381428328%` corpus WER. The compiled
alignment was also checked against 83,212 cached model responses across all
eight splits and the independent distance oracle. Every row's total edits and
WER, and every corpus WER, are unchanged; equivalent optimal alignments change
S/D/I/hit diagnostics on 379 rows. These checks establish scorer parity using
fixed predictions; evaluating a model requires separate inference rollouts.
