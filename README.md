# GutBrainIE 2026 Baselines and Evaluation

> **Research use only. Not a medical device.**
> This software was developed in the HEREDITARY project as a research prototype. It has not been assessed or certified under the EU Medical Device Regulation (EU) 2017/745. It must not be used to diagnose, treat, monitor or make any other decision about individual patients.

## 1. Overview

This repository provides the baseline implementation and evaluation pipeline for [GutBrainIE 2026](https://hereditary.dei.unipd.it/challenges/gutbrainie/2026/), Task 6 of the [BioASQ Lab](https://www.bioasq.org/) at [CLEF 2026](https://clef2026.clef-initiative.eu/). The pipeline combines named entity recognition (NER), named entity linking (NEL), and relation extraction (RE) to support four subtasks:

The full collection, comprising documents and their annotations, can be found at: [https://zenodo.org/records/23492330](https://zenodo.org/records/23492330)

Users can reproduce the complete baseline, generate predictions with the released fine-tuned models, or evaluate the supplied predictions and their own outputs.

If you use this benchmark fully or in part, please cite the following paper:

```bibtex
@inproceedings{BioASQ2026taskGutBrainIE,
   title          = {{Overview of GutBrainIE@CLEF 2026: Gut-Brain Interplay Information Extraction}},
   author         = {Martinelli, Marco and Bonato, Vanessa and Di Nunzio, Giorgio Maria and Faggioli, Guglielmo and Ferro, Nicola and Irrera, Ornella and Kantz, Benedikt and Marchesin, Stefano and Menotti, Laura and Merlo, Simone and Michelotto, Riccardo and Tussardi, Gaia and Vezzani, Federica and Waldert, Peter and Silvello, Gianmaria},    
   year           = 2026,
   booktitle    = {{CLEF 2026 Working Notes}},
   editor       = { Sánchez Salido, Eva  and Barrón-Cedeño, Alberto and García Seco de Herrera, Alba and MacAvaney, Sean and Struß, Julia Maria}
}
```

## 2. Intended use

**Primary intended use.** Reproducible development and evaluation of systems that recognize biomedical entity mentions, link them to concepts in reference resources, and extract relations from scientific literature on the gut–brain axis. The baseline provides a reference for comparing systems across the four GutBrainIE 2026 subtasks.

**Intended users.** Biomedical NLP researchers, data scientists, software developers, and clinicians conducting research on literature analysis.

**Out-of-scope uses.** This release must not be used for:

- clinical diagnosis, prognosis, treatment, monitoring, or other decisions about individual patients;
- decisions about individuals, such as insurance, employment, or access to care;
- re-identification of annotators or other individuals, including linkage of pseudonymized annotator IDs to identities;
- processing identifiable patient records or other personal records as part of the documented benchmark workflow.

Performance on other languages, domains, full-text articles, or data types has not been established. Such research requires separate evaluation. Extracted entities, concept links, and relations require source verification and must not be interpreted as validated clinical recommendations or evidence of causation.

## 3. Data provenance

The baseline uses the GutBrainIE 2026 collection of 5,000+ PubMed titles and abstracts concerning gut microbiota, Parkinson’s disease, and mental health. No full-text articles or restricted-access documents are included in the collection. Source-text reuse remains subject to applicable licensing terms. Corpus statistics and annotation guidelines are available on the [challenge website](https://hereditary.dei.unipd.it/challenges/gutbrainie/2026/).

The collection is available through the [GutBrainIE 2026 benchmark archive](https://zenodo.org/records/23492330). 

The software relies on publicly available libraries and models, including GLiNER/NuNER Zero and ATLOP. Their respective licenses and model terms continue to apply. The baseline does not require proprietary model APIs.

Concept-linking annotations contain URIs, rather than a redistribution of the underlying reference vocabularies. The NEL preparation workflow retrieves and caches descriptive information separately. Before sharing those caches, verify the applicable provider terms. Access to a URI does not itself establish permission to redistribute its associated content.

## 4. Method

- **Algorithm or architecture.** NER uses [GLiNER](https://github.com/urchade/GLiNER), initialized from [NuNER Zero](https://huggingface.co/numind/NuNER_Zero), and fine-tuned on the Gold, Silver, and Silver 2025 collections. NEL first matches mention text and entity label against concept-linked annotations, then tries mention text alone, and finally applies semantic similarity using PubMedBERT embeddings through `txtai`. Unresolved mentions receive `NA`. RE uses [ATLOP](https://github.com/wzhouad/ATLOP), with Gold, Silver, and Silver 2025 as manually annotated sets and Bronze as the distantly supervised set. Merging the predictions produces mention-level and concept-level relations.
- **Training setting.** Centralized, local training and inference. Scripts and notebooks operate on supplied files and require no interactive text input or proprietary model API. Preparing NEL resources may contact external terminology providers, and pretrained models may be downloaded before local use.

### Setup

Clone the repository and work from its root directory:

```bash
git clone https://github.com/MMartinelli-hub/GutBrainIE_2026_Baseline.git
cd GutBrainIE_2026_Baseline
```

Use an isolated Python environment, for example with Conda, and install the dependencies in `requirements.txt`:

```bash
pip install -r requirements.txt
```

For the CUDA 11.8 PyTorch setup used in the original instructions, install compatible packages in that environment:

```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Replace the empty `Annotations` and `Articles` folders with the corresponding GutBrainIE 2026 folders. Configure paths in the scripts and notebooks before running them. Run notebooks from their containing directories unless you adjust relative paths. The RE launchers use Bash; on Windows, use a suitable Bash environment or adapt the commands to PowerShell syntax.

### Quick start: evaluation and validation

1. To evaluate the supplied development predictions in `Eval/org_*`, update the prediction paths in `Eval/evaluate.py` to those files. Baseline test predictions are provided under `TestData/BaselinePredictions`.
2. Set `GROUND_TRUTH_PATH` to the reference annotations for the selected split and enable only the subtasks being evaluated.
3. From `Eval/`, run:

   ```bash
   python evaluate.py
   ```

For your own predictions, use the required 2026 fields listed in Section 1 and update the evaluator paths accordingly.

To validate prediction files, configure `PREDICTIONS_PATH_6_1_1`, `PREDICTIONS_PATH_6_1_2`, `PREDICTIONS_PATH_6_2_1`, and `PREDICTIONS_PATH_6_2_2` and the corresponding `TEST_*` flags in `Eval/validate_submission_files.py`. From `Eval/`, run:

```bash
python validate_submission_files.py
```

For a challenge submission, place each run in a folder named `<teamID>_<taskID>_<runID>[_<systemDesc>]`, using `T611`, `T612`, `T621`, or `T622`. Each run folder must contain a `.json` prediction file and a `.meta` file with the same basename as the folder. From the repository root, validate the parent folder with:

```bash
python Eval/validate_submission_folder.py /path/to/submission
```

Validation checks format and structure; it does not establish prediction accuracy. Refer to the [official submission instructions](https://hereditary.dei.unipd.it/challenges/gutbrainie/2026/) for the complete requirements.

### Prepare training data

1. Run [`Utils/annotations_to_gliner_format.ipynb`](Utils/annotations_to_gliner_format.ipynb) to produce NER data in `Train/NER/data`.
2. Run [`Utils/annotations_to_atlop_format.ipynb`](Utils/annotations_to_atlop_format.ipynb) to produce RE data in `Train/RE/data`.
3. Prepare the linking resources as described below.

### Fine-tune the NER model

In `Train/NER/gliner_interface.py`, configure the pretrained model, Gold/Silver/Silver 2025 training collections, confidence threshold, training and prediction flags, batch size, and training/evaluation steps. From `Train/NER`, run:

```bash
python gliner_interface.py
```

Evaluation checkpoints are saved in `Train/NER/logs`, and the final fine-tuned model is saved in `Train/NER/outputs`.

### Prepare NEL resources

Run [`Train/NEL/definitions/generate_definitions.ipynb`](Train/NEL/definitions/generate_definitions.ipynb) from `Train/NEL/definitions` after checking its input paths and provider access requirements. Where UMLS access is required, configure your own credentials for the helper `find_uri_definitions.py` without committing them or sharing credential-bearing files.

The notebook loads the concept URI list, gathers mention-to-concept information from the permitted annotation collections, retrieves external concept descriptions, and creates merged resources for similarity matching. Outputs include:

- `Train/NEL/definitions/merged_uri_definitions.json`
- `Train/NEL/definitions/split_uri_definitions.json`
- `Train/NEL/definitions/id_to_uri.json`

Update references to older collection paths, including any `platinum_quality` path, to the corresponding 2026 resources. When evaluating on the development set, exclude development annotations from both definition generation and exact-match dictionaries. Never use test annotations to construct linking resources for test evaluation. Development annotations may be incorporated for a final test run only where the challenge rules permit; disclose that choice when reporting results.

The linking notebook [`Train/NEL/entity_linker.ipynb`](Train/NEL/entity_linker.ipynb) builds lookup dictionaries from Gold and Silver annotations and indexes descriptive text using `neuml/pubmedbert-base-embeddings` through `txtai`. It applies the following priorities:

1. Exact match on lowercased mention text and entity label.
2. Exact match on lowercased mention text, ignoring the label.
3. The highest-ranked semantic-similarity candidate.
4. `NA` when no URI is available through those routes.

No additional task-specific PubMedBERT fine-tuning is required for this linking baseline. A GPU is recommended for embedding computation and similarity search. The pretrained encoder is downloaded when needed, while embedding computation runs locally.

### Train the RE model

1. In `Train/RE`, configure and run `compose_training_sets.py`. Use Gold, Silver, and Silver 2025 as manually annotated sets and Bronze as the distantly supervised set. It produces `data/train_annotated.json` and `data/train_distant.json`.
2. Check that `data/meta/rel2id.json` covers the training relations and that its relation count matches `num_class` in `atlop_finetune.sh`.
3. Configure training parameters and run:

```bash
bash atlop_finetune.sh
```

### Use the released fine-tuned models

To skip NER and RE training:

| Component | Released resource | Setup |
|---|---|---|
| NER | `BaselineModels/NER.zip` | Extract the contents into `Train/NER/outputs`; update the model path in `gliner_interface.py`; set `finetune_model` to `False` and `generate_predictions` to `True`. |
| NEL | Lookup resources and pretrained PubMedBERT embeddings | Generate the URI resources, configure the NER input path, and execute the linking notebook. No separately fine-tuned linking checkpoint is required. |
| RE | `BaselineModels/RE.zip` | Extract the contents into `Train/RE/outputs`; update `load_path` and `load_checkpoint` in `atlop_generate_predictions.sh`. |

### Generate NER predictions

1. Set `generate_predictions` to `True` in `Train/NER/gliner_interface.py` and run it. The original workflow saves raw predictions to `Predictions/NER/predicted_entities.json`; check the configured output path for your split.
2. Run [`Utils/NER_predictions_to_evaluation_format.ipynb`](Utils/NER_predictions_to_evaluation_format.ipynb) to produce the evaluation-format entities.
3. Run [`Utils/NER_predictions_to_atlop_format.ipynb`](Utils/NER_predictions_to_atlop_format.ipynb) to produce `Train/RE/data/predicted_entities_atlop_format.json` for RE inference.

### Generate NEL predictions

From `Train/NEL`, execute [`entity_linker.ipynb`](Train/NEL/entity_linker.ipynb) after preparing definitions and updating all prediction paths for the selected split. The checked notebook uses `Predictions/NER/predicted_entities_dev_eval_format.json` as its development input and writes linked predictions to `Predictions/NEL/predicted_entities_dev.json`.

Ensure that the same mention spans and ordering are used by NER, NEL, and the ATLOP conversion. Inconsistent entity indices can attach relation arguments to the wrong mentions.

### Generate RE predictions and merge outputs

1. From `Train/RE`, run:

   ```bash
   bash atlop_generate_predictions.sh
   ```

2. Check the output path configured by the launcher. If predictions remain at `Train/RE/outputs/results.json`, move them to the path expected by the merge notebook, for example `Predictions/RE/predicted_relations_dev.json` for the development workflow.
3. Run [`Utils/merge_predictions_to_evaluation_format.ipynb`](Utils/merge_predictions_to_evaluation_format.ipynb). Configure its NER, NEL, and RE input paths and output destinations consistently. The checked development configuration produces:

   - `Predictions/predictions_dev_eval_format.json`
   - `Eval/teamID_NER_runID_systemDesc.json`
   - `Eval/teamID_NERD_runID_systemDesc.json`
   - `Eval/teamID_MENTION_LEVEL_RE_runID_systemDesc.json`
   - `Eval/teamID_CONCEPT_LEVEL_RE_runID_systemDesc.json`

These are evaluator filenames. For challenge submission, package and name the outputs using the official `T611`, `T612`, `T621`, and `T622` run-folder conventions. The 2025 `T61`/`T623` filenames do not describe the 2026 task structure.

## 5. Performance

- **Evaluation data.** The official challenge website reports results on the GutBrainIE 2026 development and held-out test collections. The repository provides development predictions in `Eval/org_*` and test predictions under `TestData/BaselinePredictions`. These are literature-based benchmark evaluations, not clinical validation.
- **Metrics and results.** Precision, recall, and F1 are evaluated with micro- and macro-averaging. The tables below reproduce the baseline results published on the [challenge website](https://hereditary.dei.unipd.it/challenges/gutbrainie/2026/) under “Baseline Results.” Values are on a 0–1 scale.

### Test set

| Subtask | Micro precision | Micro recall | Micro F1 | Macro F1 |
|---|---:|---:|---:|---:|
| 6.1.1 — NER | 0.7782 | 0.8221 | 0.7996 | 0.7267 |
| 6.1.2 — NERD | 0.4281 | 0.4522 | 0.4398 | 0.3916 |
| 6.2.1 — Mention-level RE | 0.4462 | 0.3453 | 0.3893 | 0.3003 |
| 6.2.2 — Concept-level RE | 0.1409 | 0.1292 | 0.1348 | 0.0966 |

### Development set

| Subtask | Micro precision | Micro recall | Micro F1 | Macro F1 |
|---|---:|---:|---:|---:|
| 6.1.1 — NER | 0.7710 | 0.8278 | 0.7984 | 0.7438 |
| 6.1.2 — NERD | 0.5744 | 0.6168 | 0.5949 | 0.6102 |
| 6.2.1 — Mention-level RE | 0.4206 | 0.3652 | 0.3910 | 0.2657 |
| 6.2.2 — Concept-level RE | 0.2281 | 0.2227 | 0.2254 | 0.1878 |

Reproduced scores depend on the selected split, configuration, software versions, random seed, and linking resources. Record these choices and whether development annotations were incorporated into any final training or lookup resource.

- **Results by subgroup.** No patient-level evaluation by age, sex, country, or clinical site is reported. Entity and relation categories are benchmark labels rather than demographic groups. Confidence intervals are not reported in the cited baseline tables.

## 6. Known limitations and risks

- **Generalizability.** The baseline was evaluated on English PubMed titles and abstracts selected for the gut–brain domain. Performance on other languages, domains, full-text articles, clinical notes, or individual patient records has not been established.
- **Known biases and confounders.** The collection reflects its retrieval queries, publication coverage, and annotation schema. Annotation quality varies across the Platinum, Gold, Silver, and Bronze tiers; Bronze annotations are automatically generated and were not manually revised. Uneven entity and relation frequencies may affect performance on less frequent categories.
- **Failure modes.** NER can miss entities, assign incorrect types, or predict inaccurate boundaries. These errors propagate to RE, which can omit relations or assign incorrect predicates and arguments. Extracted associations do not establish causation or clinical efficacy, and outputs require verification against the original text.

## 7. Responsible use and reporting

- **Terms of use.** Use this release for research in accordance with the repository license, the terms of the pretrained and fine-tuned models, and the applicable source-data licenses. The research-only notice describes the intended use and clinical limitations; it does not replace the applicable license terms.
- **Contact for misuse or vulnerability reports.** Email [martinell2@dei.unipd.it](mailto:martinell2@dei.unipd.it). General questions, reproducibility issues, and proposed improvements may also be submitted through [GitHub Issues](https://github.com/MMartinelli-hub/GutBrainIE_2025_Baseline/issues). Do not post sensitive data or credentials in public issues.

## 8. Acknowledgement

Funded by the European Union under grant agreement No 101137074 (HEREDITARY). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Health and Digital Executive Agency (HaDEA). Neither the European Union nor the granting authority can be held responsible for them.
