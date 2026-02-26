# 📱 Akıllı Telefon Kullanımı, Verimlilik ve Stres Seviyesi Analizi

Bu proje, bireylerin günlük akıllı telefon kullanım süreleri, uyku düzenleri ve meslek gruplarının **Stres Seviyesi (Stress Level)** ve **İş Verimliliği (Work Productivity)** üzerindeki etkilerini Makine Öğrenmesi (Machine Learning) algoritmaları kullanarak analiz etmeyi amaçlamaktadır.

## 📊 Proje Hakkında

Proje kapsamında **50.000 satırlık** bir veri seti (`Smartphone_Usage_Productivity_Dataset_50000.csv`) kullanılmıştır. Çalışma iki temel aşamadan oluşmaktadır:

1.  **Keşifçi Veri Analizi (EDA):** Veriler arasındaki gizli ilişkileri ve dağılımları görselleştirme.
2.  **Sınıflandırma Modelleri Eğitimi:** Stres seviyesini tahmin etmek için çeşitli algoritmaların performanslarının karşılaştırılması.

## 📈 Veri Görselleştirme (EDA)

Veri setinin yapısını anlamak için `Seaborn` ve `Matplotlib` kullanılarak aşağıdaki görselleştirmeler yapılmıştır:

* **Korelasyon Matrisi:** Stres, telefon kullanımı, uyku, verimlilik ve yaş gibi 5 temel özellik arasındaki ilişkilerin ısı haritası.
* **Stres Dağılımı:** Meslek ve cinsiyete göre ortalama stres seviyelerinin analizi (farkları belirginleştirmek için Y ekseni sınırlandırılmıştır).
* **Ekran Süresi:** Cinsiyete göre günlük telefon kullanım süresinin dağılımı (KDE destekli histogram).
* **Yoğunluk Analizi:** Uyku süresi ile verimlilik skoru arasındaki ilişkinin yoğunluk grafiği (KDE Plot).

## 🤖 Model Eğitimi ve Deneyler

Model eğitimi için iki farklı Python betiği (`egitim_1.py` ve `egitim_2.py`) çalıştırılmıştır. Stres seviyesi (1-10 arası) tahmin edilmeye çalışıldığı için bu bir **çok sınıflı (multi-class)** sınıflandırma problemidir.

> **Not:** Veri setindeki sınıfların karmaşıklığı nedeniyle modeller ortalama **~%10** civarında bir doğruluk (accuracy) üretmiştir. Bu da 10 sınıflı zorlu ve dengeli dağılmış bir veri setinde beklenen bir durumdur.

### 🧪 Deney 1: `egitim_1.py`
* **Eğitim Seti:** %90
* **Test Seti:** %10

| Model Adı | Doğruluk (Accuracy) |
| :--- | :--- |
| Logistic Regression | %10.10 |
| Random Forest | %10.20 |
| **Gradient Boosting** | **%10.32 (En Başarılı)** |
| CatBoost | %10.22 |
| LightGBM | %9.90 |

### 🧪 Deney 2: `egitim_2.py`
* **Eğitim Seti:** %80
* **Test Seti:** %20
* *Amaç:* Test verisi artırıldığında modellerin genelizasyon (genelleme) yeteneğinin test edilmesi.

| Model Adı | Doğruluk (Accuracy) |
| :--- | :--- |
| Logistic Regression | %10.25 |
| Random Forest | %9.99 |
| **Gradient Boosting** | **%10.57 (En Başarılı)** |
| CatBoost | %10.39 |
| LightGBM | %9.12 |

## 🛠️ Kullanılan Teknolojiler
Bu projede aşağıdaki Python kütüphaneleri kullanılmıştır:
* **Veri İşleme:** Pandas, NumPy
* **Görselleştirme:** Matplotlib, Seaborn
* **Makine Öğrenmesi:** Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)
* **Gelişmiş Algoritmalar:** XGBoost, LightGBM, CatBoost
