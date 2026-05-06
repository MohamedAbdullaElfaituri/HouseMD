# House M.D. NLP Tanı Tahmin Projesi

## Proje Özeti

Bu proje, House M.D. dizisinden hazırlanmış replik ve vaka bilgilerini kullanarak verilen bir vaka metni için olası tanı etiketlerini tahmin eden bir doğal dil işleme çalışmasıdır. Çalışma kapsamında veri seti temizlenmiş, metin alanları modele uygun hale getirilmiş, birden fazla metin sınıflandırma modeli karşılaştırılmış ve en başarılı model web arayüzünde kullanılmak üzere kaydedilmiştir.

Proje eğitim amaçlıdır. Üretilen tahminler gerçek klinik tanı, tedavi kararı veya tıbbi öneri olarak kullanılmamalıdır.

## Amaç

Projenin temel amacı, vaka metni ve yardımcı klinik alanlardan yararlanarak `correct_prediction` alanındaki tanı etiketini tahmin etmektir. Bu amaç doğrultusunda proje aşağıdaki adımları kapsar:

- Veri setini okuyup temel kalite kontrollerini yapmak.
- Türkçe karakterleri koruyarak metinleri temizlemek ve normalleştirmek.
- Replik, semptom, test, ilaç, prosedür, organ ve meta bilgileri tek bir model girdisine dönüştürmek.
- Farklı sınıflandırma algoritmalarını aynı veri üzerinde karşılaştırmak.
- En iyi modeli kaydedip kullanıcı dostu bir tahmin arayüzü oluşturmak.

## Kullanılan Veri Seti

Veri seti `DATA/Last_HouseMD_DataSet.csv` dosyasında yer almaktadır. CSV dosyası `;` ayracı ile okunmuştur. Tüm alanlar metin olarak alınmış, boş satırlar ve kullanılamayan hedef etiketleri temizlenmiştir.

Model girdisinde kullanılan başlıca alanlar:

- `text`: Replik veya vaka metni
- `Symptom`: Semptom bilgisi
- `Test`: Test veya tetkik bilgisi
- `Drug`: İlaç bilgisi
- `Procedure`: Prosedür bilgisi
- `Organ`: Organ bilgisi
- `speaker`: Konuşmacı
- `Intent`: Konuşma niyeti
- `diagnosis_stage`: Tanı aşaması
- `Emotion`: Duygu bilgisi
- `Sarcasm`: Sarkazm bilgisi
- `medical_entities`: Metinden çıkarılmış tıbbi varlıklar

Hedef değişken `correct_prediction` alanıdır. Ayrıca `model_prediction` alanı veri sızıntısı riski taşıdığı için model girdisine dahil edilmemiştir.

## Veri Ön İşleme

Ön işleme aşamasında şu işlemler uygulanmıştır:

- Metinler küçük harfe çevrilmiştir.
- URL, gereksiz noktalama ve özel karakterler temizlenmiştir.
- Türkçe karakterler korunmuştur.
- Boş veya geçersiz hedef etiketleri veri setinden çıkarılmıştır.
- Farklı yazılmış bazı tanı etiketleri ortak bir etikete dönüştürülmüştür.
- Çok az örneğe sahip sınıflar filtrelenmiştir.
- Aynı sezon ve bölüm içerisindeki semptom, test, ilaç, prosedür, organ ve tıbbi varlık bilgilerinden kısa bir vaka bağlamı oluşturulmuştur.

Bu işlemlerden sonra modelleme için 4.415 satır ve 80 tanı sınıfı kullanılmıştır.

## Özellik Çıkarımı

Model girdisi `model_text` adlı tek bir metin alanında birleştirilmiştir. Bu alan; replik metni, klinik alanlar, meta bilgiler ve bölüm bazlı vaka bağlamından oluşur.

Metinlerden sayısal özellik çıkarmak için `TfidfVectorizer` kullanılmıştır. TF-IDF ayarları genel olarak şu şekildedir:

- Kelime tabanlı TF-IDF
- 1, 2 ve 3 kelimelik n-gramlar
- Türkçe durak kelimelerin çıkarılması
- En fazla 90.000 özellik
- `sublinear_tf=True`

## Modelleme Yöntemi

Çalışma çok sınıflı metin sınıflandırma problemi olarak ele alınmıştır. Veri seti yüzde 80 eğitim, yüzde 20 test olacak şekilde ayrılmıştır. Sınıf dağılımı dengesiz olduğu için dengeleme yalnızca eğitim verisi üzerinde uygulanmıştır. Test verisi gerçek dağılımı korumuştur.

Karşılaştırılan modeller:

| Model | Yaklaşım |
| --- | --- |
| ComplementNB word_tfidf balanced_train_max | Complement Naive Bayes |
| PassiveAggressive word_tfidf balanced_train_max | Passive-Aggressive Linear Classifier |
| SGD modified_huber word_tfidf balanced_train_max | SGD Modified Huber Linear Classifier |

En iyi model `test_macro_f1` metriğine göre seçilmiştir.

## Deney Sonuçları

Modelleme sonucunda en başarılı model `SGD modified_huber word_tfidf balanced_train_max` olmuştur.

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| SGD Modified Huber | 0.9830 | 0.9848 | 0.9823 | 0.9831 |
| Passive-Aggressive | 0.9796 | 0.9722 | 0.9761 | 0.9781 |
| Complement Naive Bayes | 0.9558 | 0.9846 | 0.9711 | 0.9571 |

Özet veri istatistikleri:

| Ölçüt | Değer |
| --- | ---: |
| Ham satır sayısı | 7.282 |
| Temizleme sonrası geçerli satır | 5.415 |
| Nadir sınıflardan çıkarılan satır | 1.000 |
| Modelde kullanılan satır | 4.415 |
| Sınıf sayısı | 80 |
| Eğitim satırı | 3.532 |
| Eğitim dengeleme sonrası satır | 27.040 |
| Test satırı | 883 |
| Minimum sınıf eşiği | 20 |

Sonuçlara göre TF-IDF özellikleri ve lineer sınıflandırıcılar bu veri setinde yüksek başarı göstermiştir. En iyi model hem genel doğruluk hem de sınıflar arası dengeyi dikkate alan Macro F1 metriğinde başarılıdır.

## Kaydedilen Model

Eğitim sonucunda seçilen model aşağıdaki dosyaya kaydedilmiştir:

```text
models/best_housemd_diagnosis_model.joblib
```

Model paketi içinde seçilen pipeline, aday modeller, sınıf listesi, test sonuçları, veri özeti ve sınıflandırma raporu bulunmaktadır. Web arayüzü bu dosyayı doğrudan yükleyerek tahmin üretir.

## Web Arayüzü

`app.py` dosyası, Python'un standart `http.server` altyapısını kullanarak basit bir web arayüzü sunar. Kullanıcı vaka metni ve klinik alanları doldurduktan sonra model en olası tanı etiketlerini sıralı şekilde döndürür.

Arayüzün sunduğu temel özellikler:

- Vaka metni girişi
- Semptom, test, ilaç, prosedür ve organ alanları
- Konuşmacı, niyet, tanı aşaması, duygu ve sarkazm alanları
- En olası 3, 5 veya 10 tanı sonucunu gösterme
- Model ve veri özet metriklerini görüntüleme

## Kurulum ve Çalıştırma

Projeyi çalıştırmak için Python 3.10 veya üzeri bir sürüm önerilir.

1. Sanal ortam oluşturun:

```bash
python -m venv .venv
```

2. Sanal ortamı etkinleştirin:

```bash
.\.venv\Scripts\activate
```

3. Gerekli paketleri kurun:

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

4. Web arayüzünü başlatın:

```bash
python app.py
```

5. Tarayıcıdan aşağıdaki adrese gidin:

```text
http://127.0.0.1:8501
```

## Proje Dosya Yapısı

```text
NLP_Project/
+-- DATA/
|   +-- Last_HouseMD_DataSet.csv
+-- models/
|   +-- best_housemd_diagnosis_model.joblib
+-- app.py
+-- HouseMD_NLP_Modelleme.ipynb
+-- .gitignore
+-- README.md
```

## Notebook İçeriği

`HouseMD_NLP_Modelleme.ipynb` dosyası modelleme sürecinin ayrıntılı halini içerir. Notebook içerisinde:

- Veri seti okuma ve kalite analizi
- Hedef etiket temizleme
- Metin normalizasyonu
- Vaka bağlamı oluşturma
- Eğitim/test ayrımı
- Eğitim verisi dengeleme
- Model karşılaştırması
- Karışıklık matrisi ve hata analizi
- En iyi modelin kaydedilmesi

adımları bulunmaktadır.

## Değerlendirme ve Limitasyonlar

Model test setinde yüksek başarı elde etmiştir; ancak sonuçlar veri setinin yapısı ve sınıf dağılımı ile sınırlıdır. Nadir tanı sınıfları modelden çıkarıldığı için model yalnızca yeterli örneğe sahip sınıflar üzerinde tahmin üretir. Ayrıca eğitim verisi House M.D. dizisinden geldiği için gerçek klinik vakaları temsil ettiği varsayılamaz.

Bu nedenle modelin çıktıları yalnızca doğal dil işleme ve makine öğrenmesi eğitimi kapsamında değerlendirilmelidir.

## Sonuç

Bu projede House M.D. veri seti kullanılarak tanı etiketi tahmini yapan bir NLP modeli geliştirilmiştir. TF-IDF tabanlı metin özellikleri ve SGD Modified Huber sınıflandırıcısı ile en iyi sonuç elde edilmiştir. Kaydedilen model, `app.py` ile sunulan web arayüzü üzerinden kullanıcı girdilerine göre en olası tanı etiketlerini listeleyebilmektedir.
