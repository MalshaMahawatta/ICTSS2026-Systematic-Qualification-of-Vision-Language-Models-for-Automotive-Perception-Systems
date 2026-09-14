# Supplementary Materials: Qualification of Vision-Language Models for Automotive Perception

This repository contains the data, model responses, analysis scripts, and run-time
monitoring artifacts associated with the paper:

> M. A. M. Dona, K. Rokanas, A. Säfström, K. Ronanki, and C. Berger,
> “Towards Systematic Qualification of Vision-Language Models for Automotive
> Perception Systems,” ICTSS 2026.

The study evaluates GPT-4o, GPT-4.5 Preview, and LLaVA on traffic-object
identification using front-camera images derived from the nuScenes dataset. It
also evaluates a SelfCheckGPT-inspired run-time monitor using LLaVA as the
captioner and GPT-4o as the consistency checker.

## Repository contents

| Path | Description |
| --- | --- |
| `labels_processed.json` | Human annotations and environmental metadata. |
| `model_outputs_gpt-4o.json` | Single-response GPT-4o outputs. |
| `model_outputs_gpt-4.5-preview.json` | Single-response GPT-4.5 Preview outputs. |
| `model_outputs_llava.json` | Single-response LLaVA outputs. |
| `compare.py` | Multiset comparison and statistical analysis. |
| `llava_captions.json` | One primary and four sampled LLaVA responses per image. |
| `selfcheck_results.json` | First SelfCheckGPT-style execution. |
| `selfcheck_results_run2.json` | Second SelfCheckGPT-style execution. |
| `llava_selfcheckgpt_prompter.py` | Generates repeated LLaVA responses. |
| `selfcheckgpt_checker_gpt4o.py` | Uses GPT-4o to judge response consistency. |
| `selfcheckgpt_evaluator.py` | Evaluates consistency judgments against labels. |
| `label.py` | Tkinter application used to inspect and annotate images. |

## Data format

Ground-truth entries use repeated object names to encode cardinality:

```json
{
  "id": "1.jpg",
  "objects": ["car", "car", "car"],
  "weather": "clear",
  "dark": false
}
```

Model-output entries contain the corresponding comma-separated response:

```json
{
  "id": "1.jpg",
  "raw_output": "car, car"
}
```

The object ontology consists of `car`, `bus`, `truck`, `bicycle`, `motorcycle`,
`pedestrian`, `traffic light red`, `traffic light green`, `traffic light yellow`,
and `speed bump`.

## Images

The nuScenes images are **not distributed in this repository**. To rerun model
inference, the scripts expect 1,026 locally curated JPEG files in a Git-ignored
directory named `saved/`, with filenames matching the `id` fields in
`labels_processed.json`.
Researchers must obtain nuScenes separately, accept its Dataset Terms, and
reconstruct the study subset using the corresponding source data. The archived
folder does not contain a public mapping from its numeric filenames to nuScenes
sample-data tokens, so that mapping must be supplied separately for independent
reconstruction.

## Installation

Python 3.9 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The OpenAI-backed scripts require an API key supplied through the environment:

```bash
export OPENAI_API_KEY="your-key"
```

LLaVA experiments require [Ollama](https://ollama.com/) running locally. The
archived experiment used the `llava:latest` model identified in the paper as
`8dd30f6b0cb1`.

## Reproducing the workflows

Generate one response per image with an OpenAI model:

```bash
python3 prompter.py gpt-4o
python3 prompter.py gpt-4.5-preview
```

Generate one response per image with LLaVA through Ollama:

```bash
python3 llava_prompter.py llava
```

Run the direct model comparison and statistical tests:

```bash
python3 compare.py
```

Generate five LLaVA responses per image, perform the GPT-4o consistency check,
and evaluate the checker against ground truth:

```bash
python3 llava_selfcheckgpt_prompter.py --model llava --samples 5
python3 selfcheckgpt_checker_gpt4o.py
python3 selfcheckgpt_evaluator.py
```

These commands generate or overwrite result JSON and figure files. Generated
comparison results and plots are intentionally not included because the retained
analysis implementation does not exactly reproduce the published tables.

## Scope and archive-integrity note

The attached ICTSS 2026 manuscript describes a 1,000-image daytime cohort. This
archive contains annotations and raw model outputs for those 1,000 entries
(`dark == false`) plus 26 additional entries marked as dark. The image files and
derived comparison results are not included.

The manuscript additionally describes WordNet-based synonym handling. The
archived `compare.py` implements case-sensitive comma splitting and exact string
matching, but does not contain that synonym-normalization step. Consequently,
the files in this archive should be treated as the retained experimental
artifacts; the current script and archived inputs do not reproduce every number
in the manuscript's tables exactly. No data have been altered in this repository
to force agreement with the published results.

## Data provenance and licensing

The experiment labels and outputs were derived using images from the nuScenes
dataset. The images themselves are excluded from this repository. Any separately
obtained copies remain subject to the
[nuScenes Dataset Terms](https://www.nuscenes.org/terms-of-use-commercial) and
the applicable Creative Commons license. See [DATA_LICENSE.md](DATA_LICENSE.md)
for attribution and reuse information.

No license has yet been assigned to the authors' source code or original
annotations. Public availability alone does not grant permission to reuse those
materials; add an explicit software/content license before publication if reuse
is intended.

## Citation

If you use these materials, cite the ICTSS 2026 paper above and the original
nuScenes dataset:

```bibtex
@inproceedings{caesar2020nuscenes,
  title     = {nuScenes: A Multimodal Dataset for Autonomous Driving},
  author    = {Caesar, Holger and Bankiti, Varun and Lang, Alex H. and Vora, Sourabh
               and Liong, Venice Erin and Xu, Qiang and Krishnan, Anush and Pan, Yu
               and Baldan, Giancarlo and Beijbom, Oscar},
  booktitle = {CVPR},
  year      = {2020}
}
```
