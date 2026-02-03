# GutBrainIE 2026 -- Baselines & Evaluation

This repository provides the complete pipeline for the baseline implementation and evaluation for the [GutBrainIE](https://hereditary.dei.unipd.it/challenges/gutbrainie/2026/) challenge – the sixth task of the [BioASQ](https://www.bioasq.org/) Lab @ [CLEF 26](https://clef2026.clef-initiative.eu/). We employ [GLiNER (NuNerZero)](https://huggingface.co/numind/NuNER_Zero) for Named Entity Recognition (NER), a hybrid approach combining exact matching and semantic similarity for Named Entity Linking (NEL), and [ATLOP](https://github.com/wzhouad/ATLOP) for Relation Extraction (RE).

Users have three main options:
- **Reproduce the baseline from scratch:** Run the entire pipeline — including data processing, training, prediction generation, and evaluation.
- **Test with pre-trained models:** Directly import our fine-tuned models for generating and evaluating predictions.
- **Evaluate provided outputs:** Use the supplied model outputs and directly run the evaluation as detailed in the TL;DR section.

Evaluation results on the dev set, along with detailed explanations of the employed evaluation metrics, are available at the [CHALLENGE WEBSITE](https://hereditary.dei.unipd.it/challenges/gutbrainie/2026/). 

These results are obtained by training the NER model on the gold, silver, and silver_2025 collections, while for the RE model the gold, silver, and silver_2025 collections were used as manually annotated sets, and the bronze collection as the distantly supervised annotated set.

---

## Getting Started

1. **Clone the Repository**

2. **Dataset Setup**: Replace the empty `Annotations` and `Articles` folders with the corresponding GutBrainIE dataset folders.

### Data Format Description

Annotations are provided in JSON format, for ease of use with NLP systems. Each entry in the dataset corresponds to a PubMed article, identified by its PubMed ID (PMID), and includes the following fields:

- **Metadata**: Article-related information, including title, author, journal, year, and abstract. It also includes the identifier of the annotator: expert annotators are labeled as expert_1 to expert_7, student annotators are grouped into two clusters, identified as student_A and student_B, and automatically generated annotations are labeled as distant. 

- **Entities**: An array of objects where each object represents an annotated entity mention in the article, with the following attributes:
  - `start` and `end` indices: Character offsets marking the span of the entity mention.
  - `location`: Indicates if the entity mention is located in the title or in the abstract.
  - `text_span`: The actual text span of the entity mention.
  - `label`: The label assigned to the entity mention (e.g., bacteria, microbiome).
  - `uri`: The URI of the concept assigned to the entity mention.
  

- **Relations**: An array of objects where each object represents an annotated relationship between two entity mentions in the article, with the following attributes:
  - `subject_start` and `subject_end` indices: Character offsets marking the span of the subject entity mention.
  - `subject_location`: Indicates if the subject entity mention is located in the title or in the abstract.
  - `subject_text_span`: The actual text span of the subject entity mention.
  - `subject_label`: The label assigned to the subject entity mention (e.g., bacteria, microbiome).
  - `subject_uri`: The URI of the concept assigned to the subject entity mention.
  - `predicate`: The label assigned to the relationship.
  - `object_start` and `object_end` indices: Character offsets marking the span of the object entity mention.
  - `object_location`: Indicates if the object entity mention is located in the title or in the abstract.
  - `object_text_span`: The actual text span of the object entity mention.
  - `object_label`: The label assigned to the object entity mention (e.g., bacteria, microbiome).
  - `object_uri`: The URI of the concept assigned to the object entity mention.

- **Mention-based Relations**: Relations extracted from the Relations array, formatted as mention-based tuples of `subject_text_span`, `subject_label`, `predicate`, `object_text_span`, and `object_label`.

- **Concept-based Relations**: Relations extracted from the Relations array, formatted as concept-based tuples of `subject_uri`, `subject_label`, `predicate`, `object_uri`, and `object_label`.

### Alternative Formats

For those more familiar with CSV or tabular formats, the dataset is also provided in these formats. In this case, each of the fields mentioned above is stored in a separate file:

- Metadata file
- Entities file
- Relations file
- Mention-based relations file 
- Concept-based relations file 

The CSV files use the pipe symbol (|) as a separator, while tabular files use the tab character (\t) for separation.

---

## Quick Start (TL;DR)
1. **Evaluate Baseline Predictions:**
   - Execute the script `Eval/evaluate.py` to generate evaluation results.
   - This step will evaluate the baseline predictions on the dev set contained in `Eval/org_*`.
   - Baseline predictions on the test set are contained in `TestData/BaselinePredictions`.
2. **Evaluate Your Predictions:**
   - Format your predictions in the submission format, as the files `Eval/org_*`.
   - Open the script `Eval/evaluate.py` and adjust the paths to point to your prediction files.
   - Execute the script `Eval/evaluate.py` to generate evaluation results.
3. **Validate Your Submission-ready Predictions**:
   - Execute the script `Eval/validate_submission_folder.py` to validate the structure of your submission folder.
   - Execute the script `Eval/validate_submission_files.py` to validate the structure of your submission files.
---

## Data Preparation and Training

### Data Conversion

Before training, convert your annotations into the required formats:

- **NER Conversion:**  
  Run [`Utils/annotations_to_gliner_format.ipynb`](Utils/annotations_to_gliner_format.ipynb) to convert annotations for GLiNER. This produces data in `Train/NER/data`.

- **NEL Data Preparation:**  
  Run [`Train/NEL/definitions/generate_definitions.ipynb`](Train/NEL/definitions/generate_definitions.ipynb) to generate URI definitions for entity linking. This extracts definitions from both training collections and external sources.

- **RE Conversion:**  
  Run [`Utils/annotations_to_atlop_format.ipynb`](Utils/annotations_to_atlop_format.ipynb) to convert annotations for ATLOP. This produces data in `Train/RE/data`.

### NER Fine-tuning

1. **Configure Training:**  
   Navigate to `Train/NER` and open `gliner_interface.py`. Adjust the following as needed:
   - Pre-trained model selection.
   - Confidence threshold for evaluation.
   - Flags for training vs. prediction.
   - Choice of training sets (gold, silver, silver_2025), training/evaluation steps, batch size, etc.

2. **Run Fine-tuning:**  
   Execute `gliner_interface.py`. If running on Windows, you need to launch that command while executing powershell in administrator mode.
   - **Outputs:**  
     - Evaluation checkpoints are saved in `Train/NER/logs`.
     - The final fine-tuned model is stored in `Train/NER/outputs`.

### NEL Setup

Named Entity Linking uses a two-stage approach combining exact matching with semantic similarity:

1. **Generate URI Definitions:**  
   Navigate to `Train/NEL/definitions` and run [`generate_definitions.ipynb`](Train/NEL/definitions/generate_definitions.ipynb). This notebook:
   - Extracts entity-to-URI mappings from training data
   - Retrieves definitions from external ontology sources
   - Creates merged definition files for similarity matching
   - **Outputs:** Various definition files in `Train/NEL/definitions/`

2. **Entity Linking Pipeline:**  
   The linking process is handled by [`Train/NEL/entity_linker.ipynb`](Train/NEL/entity_linker.ipynb), which:
   - Builds exact matching dictionaries from training data
   - Uses PubMedBERT embeddings for semantic similarity (GPU recommended)
   - Applies a priority-based linking strategy: exact match → similarity match → no match (NA)

### RE Training

1. **Prepare Training Data:**  
   In `Train/RE`, open and run `compose_training_sets.py` to decide which sets are “manually annotated” and which are “distantly annotated.”  
   - This script produces:
     - `Train/RE/data/train_annotated.json`
     - `Train/RE/data/train_distant.json`

2. **Configuration Check:**  
   Ensure that the number of relations in `Train/RE/data/meta/rel2id.json` match the `num_class` parameter in `atlop_finetune.sh` and that all the relations in your training set are included.

3. **Run Fine-tuning:**  
   Optionally adjust training parameters in `atlop_finetune.sh` and then run the script to fine-tune ATLOP. If running on Windows, you might need to replace the backslash (\) characters used for multi-line prompting with the backtick (\`) character and paste the command directly into powershell executed in administrator mode.

---

## Generating Predictions

### NER Predictions

1. **Generate Predictions:**  
   In `Train/NER`, set the `generate_predictions` flag to `True` in `gliner_interface.py` and run the script.  
   - The predictions are saved as `Predictions/NER/predicted_entities.json`.

2. **Convert Predictions:**  
   - Run [`Utils/NER_predictions_to_evaluation_format.ipynb`](Utils/NER_predictions_to_evaluation_format.ipynb) to convert GLiNER predictions into evaluation format, producing `Predictions/NER/predicted_entities_eval_format.json`.
   - Next, run [`Utils/NER_predictions_to_atlop_format.ipynb`](Utils/NER_predictions_to_atlop_format.ipynb) to convert these into the ATLOP format, saving the file in `Train/RE/data/predicted_entities_atlop_format.json`.

### NEL Predictions

1. **Run Entity Linking:**  
   Navigate to `Train/NEL` and execute [`entity_linker.ipynb`](Train/NEL/entity_linker.ipynb). This notebook:
   - Loads NER predictions from `Predictions/NER/baseline_predicted_entities_eval_format.json`
   - Applies exact matching using training data mappings
   - Performs semantic similarity matching for unmatched entities using PubMedBERT embeddings
   - **Output:** Linked entities saved to `Predictions/NEL/baseline_predicted_entities_eval_format.json`

   **Requirements:** GPU recommended for similarity matching (txtai + PubMedBERT).

### RE Predictions

1. **Generate Predictions:**  
   Navigate to `Train/RE` and run `atlop_generate_predictions.sh`. This loads your trained ATLOP model (as configured in the script) and outputs predictions to `Predictions/RE/predicted_relations.json`. 

    If running on Windows, you might need to replace the backslash (\\) characters used for multi-line prompting with the backtick (\`) character and paste the command directly into powershell executed in administrator mode. 
    
    If not done automatically by the script, move the `Train/RE/outputs/results.json` file produced in output to `Predictions/RE/predicted_relations.json`.

2. **Merge Predictions:**  
   Run [`Utils/merge_predictions_to_evaluation_format.ipynb`](Utils/merge_predictions_to_evaluation_format.ipynb) to combine the NER and RE outputs into the evaluation files `Predictions/predictions_eval_format.json`, `Eval/teamID_T61_runID_systemDesc.json`, `Eval/teamID_T621_runID_systemDesc.json`, `Eval/teamID_T622_runID_systemDesc.json`, `Eval/teamID_T623_runID_systemDesc.json`.

---

## Evaluation

With the merged predictions in `Eval/**.json`, navigate into `Eval/` and run:

```bash
python evaluate.py
```

This script calculates evaluation metrics for all subtasks. 
To test your own results, place them in the `Eval` folder, adjust the paths in `evaluate.py`, and run the script.

---

## Using Pre-trained Models

If you prefer to bypass the fine-tuning phase, you can download and use our pre-trained models.

### Importing the GLiNER Model

1. **Download:**  
   Get our fine-tuned GLiNER model from `BaselineModels/NER.zip`.
2. **Setup:**  
   Unzip and place the folder content in `Train/NER/outputs` and adjust the `model path` variable in `gliner_interface.py`.
3. **Generate Predictions:**  
   Run `gliner_interface.py` to produce NER predictions by setting the flags `finetune_model` to False and `generate_predictions` to True.

### Using NEL Components

The NEL pipeline doesn't require pre-trained models as it uses:
- **Exact matching:** Based on training data annotations and ontological definitions
- **Semantic similarity:** Uses pre-trained PubMedBERT via txtai (downloaded automatically)

Simply ensure you have:
1. Generated URI definitions using `generate_definitions.ipynb`
2. NER predictions available in the required format
3. GPU access recommended for optimal performance

### Importing the ATLOP Model

1. **Download:**  
   Get our trained ATLOP model from `BaselineModels/RE.zip`.
2. **Setup:**  
   Unzip and place the folder content in `Train/RE/outputs` and update the `load_path` and `load_checkpoint` in `atlop_generate_predictions.sh`.
3. **Generate Predictions:**  
   Execute `atlop_generate_predictions.sh` to obtain RE predictions.

---

## Dependencies & Environment Setup
All dependencies are listed in `requirements.txt`. For a consistent environment, we recommend using Conda. 

After initializing your environment, please run the following commands to ensure the correct installation of PyTorch with CUDA support:
```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Contributions & Contact

Feel free to open issues if you have any questions or improvements. For further inquiries, please reach out at: [martinell2@dei.unipd.it](mailto:martinell2@dei.unipd.it).
