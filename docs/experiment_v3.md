# Experiment: CPU LoRA adaptation of SmolLM2-135M

## Objective

Investigate whether targeted instruction examples improve command

responses from a small base language model on a Windows laptop.

## Configuration

- Base model: HuggingFaceTB/SmolLM2-135M

- Revision: 93efa2f097d58c2a74874c7e644dbc9b0cee75a2

- Device: CPU

- Precision: float32

- CPU threads: 2

- LoRA rank: 4

- LoRA alpha: 8

- Target modules: q_proj, v_proj

- Trainable parameters: 230,400

- Learning rate: 0.001

- Optimizer updates: 20

- Microbatch size: 1

- Gradients accumulated across the full training set per update

## Dataset change

V2 contained 12 training examples.

V3 added four Python-version paraphrases, giving 16 examples.

The shared instruction and six development questions were unchanged.

V3 started from the original base model, not the V2 adapter.

## Development results

| Category | V2 exact matches | V3 exact matches |
|---|---:|---:|
| New wording of trained command tasks | 3/4 | 4/4 |
| New unsupported company fact | 1/1 | 1/1 |
| Command task absent from training | 0/1 | 0/1 |

V3 corrected the Python-version answer to `python --version`.

The other previously successful answers remained correct.

For the unseen kernel-release question, V3 returned `kpartx -l`

instead of `uname -r`.

All six V3 answers ended with EOS.

## Training measurements

- V3 initial training loss: 2.799216

- V3 final training loss: 0.134620

- V3 training duration: 247.33 seconds

Training losses across V2 and V3 are not directly comparable because

the training examples and target-token distribution differ.

## Interpretation and limitations

Targeted paraphrases improved one observed development failure.

The adapter still failed a command task absent from training.

These are six visible development questions, not a sealed test set.

One successful uncertainty response does not establish reliable abstention.

V3 processed more training tokens than V2. This experiment does not

isolate wording diversity from additional training exposure.

Exact matching can reject valid alternatives, such as `python -V`.

Generated answers require review beyond the numeric score.

## Evidence

- results/v2_development_evaluation.json

- results/v3_development_evaluation.json

- results/v3_lora_training.json

- results/v3_data_validation.json
