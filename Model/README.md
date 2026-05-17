# Model - Multi-task Turkish Encoder Classifier

`Data Analysis/outputs/splits/{train,val,test}.csv` uzerinden calisan multi-task transformer.

## Egitilen Baslar

| Gorev | Sinif sayisi | Aciklama |
|---|---:|---|
| `intent` | 13 | Konusma niyeti |
| `emotion` | 10-12 | Duygu tonu |
| `diagnosis_stage` | 9 | Tani asamasi |

`sarcasm` artik egitime dahil edilmiyor. Onceki deneylerde sarcasm basligi dengesiz oldugu icin genel modeli bozdu.

Varsayilan encoder: `dbmdz/bert-base-turkish-cased`.
`ytu-ce-cosmos/modernbert-tr-base-1k` denendi, ancak bu veri bolmesinde test macro-F1 acisindan BERTurk'u gecmedi.

Mimari: shared encoder -> CLS/first token dropout -> 3 paralel linear head.

## Kurulum

```bash
pip install -r Model/requirements.txt
```

CUDA onerilir. CPU uzerinde calisir ama yavas.

## Egitim

```bash
python Model/train_multitask.py --epochs 8 --batch-size 16 --max-len 128 --class-weight none --fp16
```

Onemli argumanlar:

- `--model-name`: Hugging Face encoder adi. Varsayilan `dbmdz/bert-base-turkish-cased`.
- `--data-dir`: split CSV'lerin yeri. Varsayilan `Data Analysis/outputs/splits/`.
- `--out-dir`: checkpoint ve rapor ciktilari.
- `--min-class-count`: train'de bu sayidan az gorulen siniflari `diger` kovasina alir.
- `--class-weight`: `none` veya `balanced`.

## Ciktilar

- `checkpoint-*/`: epoch checkpointleri
- `final/model.bin`: final agirliklar
- `final/tokenizer*`: tokenizer dosyalari
- `label_maps.json`: kategori -> int eslesmeleri
- `model_config.json`: encoder, task listesi ve egitim ayarlari
- `test_metrics.json`: test macro-F1
- `test_classification_report.json`: sinif bazinda precision/recall/F1

## Gradio Demo

Demo (joblib tabanli, Torch/Transformers gerektirmez) icin hafif kurulum:

```bash
pip install -r requirements.txt
```

Not: `requirements.txt` (HouseMD kok dizininde) Gradio ve scikit-learn icin yeterlidir.
Eger egitim (Torch/Transformers) da yapacaksaniz bunun yerine:

```bash
pip install -r Model/requirements.txt
```

```bash
python Model/app/app.py
```

Demo varsayilan olarak `Model/runs/v2_tfidf_task_ensemble/score` altindaki
`feature_text_soft_vote` modellerini yukler. Bu modeller yoksa legacy
`Model/runs/multitask_berturk_nosarcasm_balanced` transformer modeline duser.

## V2 Pipeline

En yuksek skor odakli yeni pipeline:

```bash
python "Data Analysis/scripts/prepare_v2_pipeline.py"
python Model/baselines/v2_tfidf_task_ensemble.py --split-family both --save-models
python Model/baselines/v2_linear_model_search.py --split-family both
```

Urettigi ana dosyalar:

- `Data Analysis/outputs/cleaned_dataset_v2.csv`
- `Data Analysis/outputs/splits_score/{train,val,test}.csv`
- `Data Analysis/outputs/splits_fair/{train,val,test}.csv`
- `Data Analysis/reports/v2_label_audit_{intent,emotion,diagnosis_stage}.csv`
- `Model/runs/v2_tfidf_task_ensemble/{score,fair}/ablation_summary.csv`
- `Model/runs/v2_linear_model_search/{score,fair}/selected_task_results.csv`
- `Model/runs/v2_best_current/{score,fair}/selected_summary.json`

Son en iyi kombinasyon:

| Split | Secim | 3-task macro-F1 | Weighted-F1 |
|---|---|---:|---:|
| `splits_score` | linear search `intent` + TF-IDF soft-vote `emotion` + linear search `diagnosis_stage` | 0.4311 | 0.5711 |
| `splits_fair` | TF-IDF `intent` + linear search `emotion` + linear search `diagnosis_stage` | 0.3545 | 0.4789 |

Score split task sonuclari:

| Gorev | Secilen model | Test macro-F1 |
|---|---|---:|
| `intent` | `compact_feature_text__union_logreg_c2` | 0.4211 |
| `emotion` | `feature_text_soft_vote` | 0.3410 |
| `diagnosis_stage` | `feature_text__union_svc_c05` | 0.5313 |

BERTurk frozen embedding + linear baseline de denendi, ancak TF-IDF/linear
arama altinda kaldi: `splits_score` 3-task macro-F1 `0.3009`.

Task-specific transformer egitimi:

```bash
python Model/train_task_classifier.py --data-dir "Data Analysis/outputs/splits_score" --task intent --out-dir Model/runs/v2_task_berturk_score --model-name dbmdz/bert-base-turkish-cased --text-col feature_text --epochs 6 --batch-size 16 --oversample --augment --fp16
```

`intent`, `emotion`, `diagnosis_stage` icin ayri ayri calistirilir.

1 epoch BERTurk smoke test (`splits_score`, `feature_text`, oversample + augment)
final skor degildir; yalnizca yeni trainer'in uctan uca calistigini dogrulamak
icin kosuldu:

| Gorev | Test macro-F1 | Test weighted-F1 |
|---|---:|---:|
| `intent` | 0.2867 | 0.4777 |
| `emotion` | 0.1094 | 0.3785 |
| `diagnosis_stage` | 0.3994 | 0.3938 |
| Ortalama | 0.2652 | - |

## Etik Notlar

- Tibbi tavsiye degildir. Veri kumesi kurgusal televizyon diyalogudur.
- Etiketler tek-anotatorlu ve gurultuludur.
- House karakteri veri icinde baskindir; duygu ve sarcasm dagilimini saptirir.
- Telifli senaryolardan turetilmis olabilir; akademik kullanim disinda ham veriyi dagitmayin.
