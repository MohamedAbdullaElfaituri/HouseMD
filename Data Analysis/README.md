# Data Analysis — House MD Türkçe NLP Veri Kümesi

Bu klasör, BIM 432 Doğal Dil İşleme dersi için toplanan House MD Türkçe diyalog veri kümesinin
denetimi, temizlenmesi ve keşifsel analizini içerir.

## Klasör yapısı

```
Data Analysis/
├── README.md                       # bu dosya
├── requirements.txt                # Python bağımlılıkları
├── canonical_labels.yaml           # eşanlamlı etiket → kanonik form haritası
├── notebooks/                      # EDA Jupyter notebookları (01–09)
├── scripts/                        # tekrar kullanılabilir Python scriptleri
│   ├── clean_dataset.py            # ana temizleyici (yaml'i yükler, hepsini uygular)
│   ├── label_synonyms.py           # otomatik eşanlamlı küme keşfi
│   ├── parse_multivalue.py         # virgülle ayrılmış hücreleri parse eder
│   ├── tokenize_tr.py              # Türkçe tokenleştirme yardımcısı
│   ├── entities_to_bio.py          # medical_entities JSON → BIO CoNLL
│   └── make_splits.py              # bölüm-ayrık train/val/test bölmesi
├── figures/                        # notebooklar tarafından üretilen .png çıktılar
├── reports/
│   ├── cleaning_report.md          # uygulanan tüm temizlik kuralları
│   ├── eda_summary.md              # EDA özeti (Türkçe, 1 sayfa)
│   ├── inconsistency_log.csv       # her satırın hata günlüğü
│   └── label_clusters_*.csv        # label_synonyms.py'nin çıktıları
└── outputs/
    ├── cleaned_dataset.csv         # temizlenmiş veri (16 sütun, snake_case)
    ├── cleaned_dataset.parquet     # aynı veri, hızlı yükleme için
    ├── label_maps.json             # her etiket sütunu için kanonik → int
    ├── entities_bio.conll          # NER projesi için BIO formatı
    └── splits/                     # train.csv, val.csv, test.csv
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma sırası

1. **Etiket eşanlamlı keşfi** (sadece bir kez gerekir, sonra YAML'i elle düzenle):
   ```bash
   python scripts/label_synonyms.py
   ```
   Çıktıları `reports/label_clusters_<col>.csv` dosyalarında incele, küme başına bir kanonik form
   seçip `canonical_labels.yaml` dosyasını güncelle.

2. **Temizleme**:
   ```bash
   python scripts/clean_dataset.py
   ```
   `outputs/cleaned_dataset.csv` ve `reports/cleaning_report.md` üretir.

3. **EDA notebookları** (sırayla 01 → 09):
   ```bash
   jupyter notebook notebooks/
   ```

4. **Bölmeler**:
   ```bash
   python scripts/make_splits.py
   ```
   Bölüm bazlı (episode-disjoint) train/val/test üretir.

5. **NER stretch hedefi**:
   ```bash
   python scripts/entities_to_bio.py
   ```

## Ham veri kümesi

Konum: `../DATASET/Last_HouseMD_DataSet(Sayfa1).csv`
- 7,282 satır
- Noktalı virgül (;) ayırıcı, UTF-8 (BOM'lu)
- 16 adlandırılmış sütun + 24 boş tamamlama sütunu

## Önemli not — etiket karmaşası

İlk inceleme, etiket sütunlarındaki (`intent`, `emotion`, `diagnosis_stage`)
değerlerin proje şartnamesinde belirtilen sabit kümeden (11/8/8) ÇOK daha geniş olduğunu
ortaya çıkardı. Sınıf arkadaşları satırları farklı yaklaşımlarla etiketlemiş; her toplu yükleme
kendi serbest formlu kelime dağarcığını kullanıyor. `label_synonyms.py` ve `notebooks/09` adımı
bu kaosu çözmek için tasarlandı.
