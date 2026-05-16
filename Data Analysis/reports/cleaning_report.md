# Temizleme Raporu — House MD veri kümesi

- Ham satır sayısı: **7282** (başlık hariç)
- Temizlenmiş satır sayısı: **7245**
- Toplam tutarsızlık girdisi: **488**

## Sütun bazlı en sık sorunlar

| Sütun | Sorun | Adet |
|---|---|---:|
| `medical_entities` | JSON parse failure | 239 |
| `sarcasm` | non-binary value | 193 |
| `<row>` | exact duplicate (text + speaker + episode) | 29 |
| `text` | empty text | 8 |
| `season` | non-numeric value | 7 |
| `emotion` | unmapped label value | 2 |
| `Symptom` | mixed-case column name | 1 |
| `Test` | mixed-case column name | 1 |
| `Drug` | mixed-case column name | 1 |
| `Procedure` | mixed-case column name | 1 |
| `Intent` | mixed-case column name | 1 |
| `Sarcasm` | mixed-case column name | 1 |
| `Emotion` | mixed-case column name | 1 |
| `Organ` | mixed-case column name | 1 |
| `episode` | non-numeric value | 1 |
| `text` | 157 near-duplicates flagged (≥0.9 Jaccard) | 1 |

## Etiket kardinalitesi (temizleme sonrası)

| Sütun | Benzersiz değer | Beklenen (şartname) |
|---|---:|---:|
| `intent` | 13 | 11 |
| `emotion` | 12 | 8 |
| `diagnosis_stage` | 9 | 8 |
| `sarcasm` | 2 | 2 |
| `organ` | 683 | — |

> Not: Şartname 11/8/8/2 sınıf öngörüyor ama veri seti elle etiketlendiği için
> her sütunda çok daha fazla yüzey form var. `label_synonyms.py` bunu kümeler.
> Eşleştirilmemiş değerler `inconsistency_log.csv` dosyasına yazılır.

## Uygulanan kurallar

1. UTF-8 BOM kaldırma (utf-8-sig)
2. 24 boş tamamlama sütunu drop
3. Sütun adları → snake_case
4. Bütün metin sütunlarında trim
5. Mojibake onarımı (ftfy)
6. season/episode → Int16
7. sarcasm → Int8 (0/1)
8. Çok-değerli hücreler (`,` ile ayrılmış) listeye parse → `*_list` sütunu
9. `canonical_labels.yaml` ile etiket normalizasyonu
10. `medical_entities` JSON unescape + parse
11. Boş `text` satırlarını sil
12. Tam yinelenen satırları sil (text + speaker + episode)
13. Yakın yinelenenleri MinHash ile işaretle (≥0.9 Jaccard)