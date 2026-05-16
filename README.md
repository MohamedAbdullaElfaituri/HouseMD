# House MD Turkish NLP

This repository contains the cleaned House MD Turkish dialogue dataset, the final selected models, and the scripts needed to reproduce the preprocessing and model experiments.

The project predicts three labels from a dialogue line:

- `intent`: the purpose of the line
- `emotion`: the emotional tone
- `diagnosis_stage`: where the line belongs in the medical reasoning process

`sarcasm` was removed from the final modeling target because it was noisy and hurt the multi-task setup.

## Folder Structure

```text
DATASET/
  Last_HouseMD_DataSet(Sayfa1).csv

Data Analysis/
  scripts/
  outputs/
    cleaned_dataset.csv
    cleaned_dataset_v2.csv
    splits_score/
    splits_fair/
  reports/

Model/
  app/
  baselines/
  runs/
    v2_linear_model_search/
    v2_tfidf_task_ensemble/
    v2_best_current/

sunum/
```

`sunum/` is intentionally empty so you can put the final presentation file there.

## Install

```bash
python -m pip install -r requirements.txt
```

For old transformer experiments, use `Model/requirements.txt` instead. The final demo does not require Torch or Transformers.

## Run Demo

```bash
python Model/app/app.py
```

The demo loads the current best score-split combination:

- intent: `Model/runs/v2_linear_model_search/score/selected_intent.joblib`
- emotion: `Model/runs/v2_tfidf_task_ensemble/score/model_feature_text_soft_vote_emotion.joblib`
- diagnosis_stage: `Model/runs/v2_linear_model_search/score/selected_diagnosis_stage.joblib`

## Rebuild Data Pipeline

```bash
python "Data Analysis/scripts/clean_dataset.py"
python "Data Analysis/scripts/prepare_v2_pipeline.py"
```

These generate:

- `Data Analysis/outputs/cleaned_dataset.csv`
- `Data Analysis/outputs/cleaned_dataset_v2.csv`
- `Data Analysis/outputs/splits_score/{train,val,test}.csv`
- `Data Analysis/outputs/splits_fair/{train,val,test}.csv`

## Retrain Final Classical Models

```bash
python Model/baselines/v2_tfidf_task_ensemble.py --split-family both --save-models
python Model/baselines/v2_linear_model_search.py --split-family both
```

## Current Best Scores

Score split:

- intent macro-F1: `0.4211`
- emotion macro-F1: `0.3410`
- diagnosis_stage macro-F1: `0.5313`
- 3-task average macro-F1: `0.4311`

Fair split:

- intent macro-F1: `0.3369`
- emotion macro-F1: `0.2983`
- diagnosis_stage macro-F1: `0.4283`
- 3-task average macro-F1: `0.3545`

## Notes

- The dataset is based on fictional TV dialogue, not real clinical records.
- The model is not medical advice.
- Label noise remains, especially in `emotion`.
- The best improvement came from data cleaning, label taxonomy simplification, context/entity feature text, and task-specific linear model search.
