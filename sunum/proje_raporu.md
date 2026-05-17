# HouseMD Türkçe NLP — Proje Raporu

## 1) Özet
Bu proje, House M.D. dizisine ait Türkçe diyalog satırlarından **3 farklı etiketi** tahmin eden çok görevli (multi-task) bir sınıflandırma hattı sunar:

- **intent**: repliğin iletişimsel amacı (örn. soru, test, tedavi, hipotez)
- **emotion**: duygusal ton (örn. nötr, ciddi, kaygı, öfke)
- **diagnosis_stage**: tıbbi akıl yürütme sürecindeki aşama (örn. başlangıç, hipotez, test, tedavi)

Repo; veri temizleme/normalizasyon adımlarını, tekrar üretilebilir veri bölmelerini, klasik ML bazlı nihai modelleri ve hafif bir **Gradio demo** arayüzünü içerir.

> Not: `sarcasm` hedefi final modellemeden çıkarılmıştır (gürültülü/dengesiz olduğu için).

## 2) Veri Kümesi
- Ham veri: `HouseMD/DATASET/Last_HouseMD_DataSet(Sayfa1).csv`
- Ham satır sayısı: **7,282**
- Temizlenmiş satır sayısı: **7,245**

Veri kümesi; sezon/bölüm, konuşmacı (speaker), metin (text), bazı tıbbi varlık alanları ve hedef etiket sütunlarını içerir.

### 2.1) Veri kalitesi / sorunlar
Temizleme sırasında en sık karşılaşılan sorunlar:
- `medical_entities` alanında JSON parse hataları
- `sarcasm` alanında 0/1 dışı değerler
- Tam kopya satırlar (text + speaker + episode)
- Boş metin satırları
- Sezon/bölüm alanlarında numerik olmayan değerler

Ayrıntı: `HouseMD/Data Analysis/reports/cleaning_report.md`

## 3) Veri Temizleme ve Normalizasyon
Veri temizleme hattı, `HouseMD/Data Analysis/scripts/clean_dataset.py` ve V2 hattı için `HouseMD/Data Analysis/scripts/prepare_v2_pipeline.py` üzerinden yönetilir.

### 3.1) Uygulanan temel temizlik kuralları (özet)
- Kodlama/BOM düzeltmeleri, metin trim
- Sütun adlarının `snake_case` formatına çekilmesi
- Tip dönüşümleri (`season`, `episode`, `sarcasm`)
- Çok değerli hücrelerin parse edilmesi (`,`) → `*_list` sütunları
- Etiket eşanlamlılarının birleştirilmesi (`canonical_labels.yaml`)
- Boş metin satırlarının ve tam kopyaların temizlenmesi

### 3.2) Etiket taksonomisi
Ham veri, serbest biçimli etiketler içerdiği için etiketler kanonik sınıflara indirgenmiştir:
- Eşleme kaynağı: `HouseMD/Data Analysis/canonical_labels.yaml`
- V2 pipeline ek olarak `prepare_v2_pipeline.py` içinde daha kompakt sınıf setine map eder (örn. `diğer` kovası).

Temizleme sonrası etiket kardinalitesi örneği:
- intent: **13**
- emotion: **12**
- diagnosis_stage: **9**

## 4) Veri Bölme (Train/Val/Test)
Proje iki split ailesi üretir:

1. **score split** (`splits_score`): stratified random bölme (daha yüksek skor odaklı)
2. **fair split** (`splits_fair`): **episode-disjoint** (aynı bölümden train/test sızıntısını azaltır)

Çıktılar:
- `HouseMD/Data Analysis/outputs/splits_score/{train,val,test}.csv`
- `HouseMD/Data Analysis/outputs/splits_fair/{train,val,test}.csv`

## 5) Modelleme Yaklaşımı
Repo içinde iki ana modelleme yolu bulunur:

### 5.1) Nihai (demo) yaklaşım — Klasik ML + joblib (önerilen)
- Metin temsili: TF-IDF ve/veya özellikli şablon metinler (`feature_text`, `compact_feature_text`)
- Modeller: Logistic Regression / Linear SVC benzeri lineer sınıflandırıcılar
- Seçim: görev başına ayrı model araması + en iyi kombinasyon
- Çıktı formatı: `joblib` (Torch/Transformers gerektirmez)

Demo varsayılan olarak şu dosyaları yükler:
- intent: `HouseMD/Model/runs/v2_linear_model_search/score/selected_intent.joblib`
- emotion: `HouseMD/Model/runs/v2_tfidf_task_ensemble/score/model_feature_text_soft_vote_emotion.joblib`
- diagnosis_stage: `HouseMD/Model/runs/v2_linear_model_search/score/selected_diagnosis_stage.joblib`

### 5.2) Transformer deneyleri (opsiyonel)
- Multi-task encoder mimarisi (BERTurk vb.)
- Eğitim scriptleri: `HouseMD/Model/train_multitask.py`, `HouseMD/Model/train_task_classifier.py`
- Kurulum: `HouseMD/Model/requirements.txt`

## 6) Sonuçlar
Aşağıdaki metrikler test seti üzerinde raporlanmıştır (macro-F1):

### 6.1) En iyi kombinasyon (score split)
- intent: **0.4211**
- emotion: **0.3410**
- diagnosis_stage: **0.5313**
- 3 görev ortalama: **0.4311**

### 6.2) En iyi kombinasyon (fair split)
- intent: **0.3369**
- emotion: **0.2983**
- diagnosis_stage: **0.4283**
- 3 görev ortalama: **0.3545**

Kaynak: `HouseMD/Model/runs/all_scores_summary.csv` ve repo kök `HouseMD/README.md`.

## 7) Demo Uygulaması (Gradio)
Hafif demo, yalnızca joblib modelleri ile çalışır:

```bash
# workspace root (NLP_Project/) içinden
python -m pip install -r HouseMD/requirements.txt
python HouseMD/Model/app/app.py
```

> Önemli: joblib modelleri `scikit-learn==1.8.0` ile kaydedilmiştir. Farklı sürümler inference sırasında uyumsuzluk çıkarabilir.

## 8) Tekrar Üretilebilirlik (Repro)
V2 veri hattını ve split’leri tekrar üretmek için:

```bash
python "HouseMD/Data Analysis/scripts/clean_dataset.py"
python "HouseMD/Data Analysis/scripts/prepare_v2_pipeline.py"
```

Klasik ML modellerini yeniden eğitmek için:

```bash
python HouseMD/Model/baselines/v2_tfidf_task_ensemble.py --split-family both --save-models
python HouseMD/Model/baselines/v2_linear_model_search.py --split-family both
```

## 9) Sınırlılıklar ve Etik Notlar
- Veri, **kurgusal dizi diyaloglarıdır**; gerçek klinik kayıt değildir.
- Çıktılar **tıbbi tavsiye değildir**.
- Etiketler tek anotatörlü ve gürültülüdür (özellikle `emotion`).
- Telif/dağıtım açısından ham veri setinin paylaşımı kısıtlı olabilir; bu repo eğitim amaçlı kullanılmalıdır.

## 10) İlgili Dosyalar
- Genel proje açıklaması: `HouseMD/README.md`
- Veri analizi ve pipeline: `HouseMD/Data Analysis/README.md`
- Temizleme raporu: `HouseMD/Data Analysis/reports/cleaning_report.md`
- Skor özetleri: `HouseMD/Model/runs/all_scores_summary.csv`
- Demo uygulaması: `HouseMD/Model/app/app.py`
- Sunum PDF: `HouseMD/sunum/House_MD_Medical_NLP.pdf`
