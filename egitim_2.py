import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import warnings

# Uyarıları kapat
warnings.filterwarnings('ignore')

# 1. Veri Setini Yükleme
dosya_yolu = r"C:\Users\semai\OneDrive\Desktop\Proje_1\Smartphone_Usage_Productivity_Dataset_50000.csv"
try:
    df = pd.read_csv(dosya_yolu)
    print("Veri seti başarıyla yüklendi.")
except FileNotFoundError:
    print(f"Hata: Dosya bulunamadı -> {dosya_yolu}")
    exit()

if 'User_ID' in df.columns:
    df = df.drop('User_ID', axis=1)

# Grafik ayarları
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# DÜZELTME 1: KORELASYON MATRİSİ (Seçilen Anlamlı 5 Özellik)
# ---------------------------------------------------------------------------

# Analiz için anlamlı olabilecek 5 temel özelliği manuel olarak seçiyoruz.
# Bu özellikler genellikle birbirini etkilemesi beklenen değişkenlerdir.
secilen_ozellikler = [
    'Stress_Level',           # Hedef değişken (Stres)
    'Daily_Phone_Hours',      # Telefon kullanımı (Stresi etkiler)
    'Sleep_Hours',            # Uyku süresi (Stres ve verimliliği etkiler)
    'Work_Productivity_Score', # Verimlilik Skoru (Sonuç)
    'Age'                     # Yaş (Davranışları etkileyebilir)
]

# Eğer veri setinde 'Age' yoksa onun yerine başka bir sayısal sütun veya mevcut olanları kullanalım
mevcut_sutunlar = [col for col in secilen_ozellikler if col in df.columns]

# Eksik sütun varsa tamamlamak için sayısal sütunlardan ekleme yapalım
if len(mevcut_sutunlar) < 5:
    diger_sayisal = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for col in diger_sayisal:
        if col not in mevcut_sutunlar and len(mevcut_sutunlar) < 5:
            mevcut_sutunlar.append(col)

# Sadece bu seçili özellikler için korelasyon hesapla
df_ozel = df[mevcut_sutunlar].copy()
corr = df_ozel.corr()

plt.figure(figsize=(10, 8))
# Maske (Sadece alt üçgeni göster)
mask = np.triu(np.ones_like(corr, dtype=bool))

# Isı haritası
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='RdBu', 
            vmin=-1, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title(f'Seçilen {len(mevcut_sutunlar)} Temel Özellik Arasındaki İlişki', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.show()

# ---------------------------------------------------------------------------
# DÜZELTME 2: FARKLILIKLARI ÖNE ÇIKARAN GRAFİKLER
# ---------------------------------------------------------------------------

# Grafik 1: Meslek ve Cinsiyete Göre Stres (Y-Ekseni Zoomlu)
plt.figure(figsize=(12, 6))
sns.barplot(x='Occupation', y='Stress_Level', hue='Gender', data=df, palette='viridis', errorbar=None)
# Y eksenini daraltarak farkları büyütelim (Verinin %99'u 0-10 arasındadır ama ortalamalar yakınsa zoom yapalım)
ylim_min = df.groupby(['Occupation', 'Gender'])['Stress_Level'].mean().min() * 0.9
ylim_max = df.groupby(['Occupation', 'Gender'])['Stress_Level'].mean().max() * 1.1
plt.ylim(ylim_min, ylim_max) 
plt.title('Meslek ve Cinsiyete Göre Ortalama Stres (Farkları Görmek İçin Zoom Yapıldı)')
plt.show()

# Grafik 2: Ekran Süresi Dağılımı (Histogram ile Detay)
# Ortalama almak yerine dağılımı görmek daha iyidir
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='Daily_Phone_Hours', hue='Gender', kde=True, element="step", palette='rocket')
plt.title('Cinsiyete Göre Günlük Telefon Kullanımı Dağılımı')
plt.xlabel('Saat')
plt.show()

# Grafik 3: Verimlilik ve Uyku (Scatter - Yoğunluk)
# Eğer veri çok düzgün dağıldıysa (random), scatter yerine hexbin veya contour daha iyi gösterir
plt.figure(figsize=(10, 8))
sns.kdeplot(x=df['Sleep_Hours'], y=df['Work_Productivity_Score'], cmap="Blues", fill=True, thresh=0.05)
plt.title('Uyku Süresi ve Verimlilik Arasındaki Yoğunluk İlişkisi')
plt.show()

# ---------------------------------------------------------
# 4. Model Eğitimi (Daha Güçlü Modeller Ekleyelim)
# ---------------------------------------------------------
print("\nVeri ön işleme ve model eğitimi başlıyor...")

# Asıl veri seti üzerinde işlem yapalım
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

y = df['Stress_Level']
X = df.drop('Stress_Level', axis=1)

# Veriyi %80 eğitim, %20 test olarak ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42), # Parametreler güçlendirildi
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, iterations=200, depth=6, learning_rate=0.1, random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=42, verbose=-1)
}

print(f"\n{'Model Adı':<25} | {'Doğruluk':<20}")
print("-" * 50)

results = {}
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name:<25} | %{acc*100:.2f}")
    except Exception as e:
        print(f"{name:<25} | HATA: {e}")

if results:
    best_model = max(results, key=results.get)
    print("\n" + "-"*50)
    print(f"En Başarılı Model: {best_model} (Doğruluk: %{results[best_model]*100:.2f})")
    
    # ---------------------------------------------------------------------------
    # DÜZELTME 3: PAST GRAFİĞİ (Modellerin Doğruluk Oranları)
    # ---------------------------------------------------------------------------
    plt.figure(figsize=(10, 8))
    
    # Verileri hazırla
    labels = list(results.keys())
    sizes = [val * 100 for val in results.values()] # Yüzdeye çevir
    
    # Renk paleti
    colors = sns.color_palette('pastel')[0:len(labels)]
    
    # Pasta grafiği çizimi
    # explode: Dilimleri birbirinden hafifçe ayırmak için
    explode = [0.05] * len(labels) 
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
            startangle=140, pctdistance=0.85, explode=explode, shadow=True)
            
    # Ortasına beyaz daire ekleyerek "Donut Chart" görünümü verelim (Daha modern durur)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.title('Modellerin Doğruluk Oranları (Accuracy)', fontsize=16)
    plt.tight_layout()
    plt.axis('equal') # Dairenin tam yuvarlak olmasını sağlar
    plt.show()