# SLM Enterprise Knowledge Lab

CPU-based experiments in small language model adaptation and
retrieval-augmented generation, developed on a Windows laptop.

The project records successful results and failure cases, including
incorrect commands, unsupported answers, and limitations of exact-match
evaluation.

## Demonstrated results

- Trained a 9,515-parameter toy transformer from scratch.
- Fine-tuned SmolLM2-135M using 230,400 LoRA parameters.
- Compared base-model, zero-shot, few-shot, and adapted outputs.
- Built section-based document ingestion with source hashes and line ranges.
- Compared keyword and semantic retrieval.
- Connected live retrieval, a provisional relevance gate, and generation.
- Checked generated commands against referenced source lines.

## Selected experiment findings

| Experiment | Observed result |
|---|---|
| V2 adapter: new wording of trained tasks | 3/4 exact matches |
| V3 adapter: new wording of trained tasks | 4/4 exact matches |
| V3 adapter: unseen kernel-release task without evidence | Incorrect command |
| Same kernel question with relevant evidence | Correct: uname -r |
| Irrelevant-context control | Explicit uncertainty on three probes |
| Live session source-command verification | 2/2 generated commands matched source lines |

These are small, visible development experiments, not general accuracy
benchmarks. V3 data additions were guided by observed development errors.

## Hardware and approach

- Windows laptop
- Intel Core i7-8550U
- Approximately 32 GB RAM
- CPU execution using two PyTorch threads
- Project, model cache, and temporary files stored on E:

The NVIDIA GPU was not used for these experiments.

## Models

- Generation: HuggingFaceTB/SmolLM2-135M
- Embeddings: sentence-transformers/all-MiniLM-L6-v2

Pinned model revisions are recorded in configs/.

## Repository layout

- lessons/: executable learning exercises
- configs/: pinned model revisions
- data/: synthetic instruction datasets and curated reference documents
- results/: recorded experiment outputs
- docs/: experiment interpretation and limitations

Downloaded models, adapter weights, checkpoints, and the Python virtual
environment are excluded from Git.

## Windows setup

Run from the repository folder in PowerShell.
Python 3.12 and Git are prerequisites.

```powershell
New-Item -ItemType Directory -Force E:\slm-cache\huggingface, E:\slm-cache\pip, E:\slm-temp

$env:HF_HOME = "E:\slm-cache\huggingface"
$env:PIP_CACHE_DIR = "E:\slm-cache\pip"
$env:TEMP = "E:\slm-temp"
$env:TMP = "E:\slm-temp"

py -3.12 -m venv .venv

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "torch==2.15.0+cpu" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

requirements-lock.txt records the development environment.
Dependency installation on a separate machine has not yet been verified.
Initial model loading requires internet access.

## Reproduce the V3 adapter and live demonstration

The source datasets and pinned model configuration are included.
Run these scripts in order:

```powershell
.\.venv\Scripts\python.exe lessons\23_validate_v3_data.py
.\.venv\Scripts\python.exe lessons\24_prepare_v3_tensors.py
.\.venv\Scripts\python.exe lessons\25_train_v3_adapter.py
.\.venv\Scripts\python.exe lessons\26_evaluate_v3.py
.\.venv\Scripts\python.exe lessons\27_chunk_documents.py
.\.venv\Scripts\python.exe lessons\29_semantic_retrieval.py
.\.venv\Scripts\python.exe lessons\34_live_rag.py
```

The training script refuses to overwrite an existing V3 adapter folder.
Run training once in a fresh checkout, or choose a new output path.

In the live application, try:

- Which Linux command prints the kernel release?
- How do I find my current folder's path in Linux?
- What is Meridian's escalation telephone number?

Type exit to save the session report.

Recorded results may be overwritten by scripts that use fixed report names.
Live sessions receive timestamped filenames.

## Important limitations

- The knowledge corpus contains only five Linux command sections.
- The retrieval threshold of 0.50 was selected after inspecting development examples.
- Similarity scores are not probabilities of correctness.
- Application-controlled withholding is separate from model abstention.
- Displaying a source reference does not establish that an answer is supported.
- Source-command verification checks exact agreement, not source correctness or question relevance.
- Some diagnostic scripts contain fixed session filenames or reviews of specific observed outputs.
- Generated commands are displayed as text; the application does not execute them.
- This is an educational prototype, not a production support system.

## Experiment evidence

See docs/experiment_v3.md for the adaptation experiment.

Additional evidence includes:

- results/v3_development_evaluation.json
- results/grounded_generation.json
- results/irrelevant_context_control_reviewed.json
- results/live_answers_verified.json
