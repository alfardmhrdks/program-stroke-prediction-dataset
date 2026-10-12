import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.utils import resample

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "stroke"

DATA_PATH = os.environ.get("STROKE_CSV", "")
CANDIDATE_PATHS = [
    "healthcare-dataset-stroke-data.csv",
    "/content/healthcare-dataset-stroke-data.csv",
    "/kaggle/input/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv",
]


def garis(judul):
    print("\n" + "=" * 90)
    print(judul)
    print("=" * 90)


def tampil_atas_bawah(df, judul, n=5):
    garis(judul)
    print(f"Ukuran: {df.shape[0]} baris x {df.shape[1]} kolom\n")
    print(f"--- {n} baris teratas ---")
    print(df.head(n).round(3).to_string())
    print(f"\n--- {n} baris terbawah ---")
    print(df.tail(n).round(3).to_string())


# %% [markdown]
# ## 0. Memuat data

# %%
def muat_data():
    paths = ([DATA_PATH] if DATA_PATH else []) + CANDIDATE_PATHS
    for p in paths:
        if os.path.exists(p):
            print(f"Membaca data dari: {p}")
            return pd.read_csv(p, na_values=["N/A", "NA", "n/a", ""])
    try:  # cadangan: unduh otomatis dari Kaggle
        import kagglehub

        folder = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
        p = os.path.join(folder, "healthcare-dataset-stroke-data.csv")
        print(f"Membaca data dari: {p}")
        return pd.read_csv(p, na_values=["N/A", "NA", "n/a", ""])
    except Exception as e:
        raise FileNotFoundError(
            "File CSV tidak ditemukan. Unduh dari "
            "https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset "
            "lalu isi variabel DATA_PATH."
        ) from e


df_raw = muat_data()
df = df_raw.copy()                                                  

# %% [markdown]
# ## 1. Memahami data: info & tipe kolom

# %%
garis("1. INFO DATA (sebelum praproses)")
print("Ukuran data:", df.shape)
print("\nTipe kolom & jumlah non-null:")
df.info()
print("\nStatistik deskriptif (numerik):")
print(df.describe().round(2).to_string())

print("\nNilai unik kolom kategorikal (sebelum dibersihkan):")
KOLOM_KATEGORI_AWAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
for c in KOLOM_KATEGORI_AWAL:
    print(f"\n[{c}]")
    print(df[c].value_counts(dropna=False).to_string())

# %% [markdown]
# ## 2. Koreksi tipe kolom & penyeragaman nama kolom

# %%
garis("2. KOREKSI TIPE KOLOM")

# a) Kolom identitas bukan fitur -> dibuang (juga mencegah leakage berbasis ID)
if "id" in df.columns:
    print("Kolom 'id' dibuang (hanya nomor pengenal, bukan fitur).")
    df = df.drop(columns=["id"])

# b) Seragamkan nama kolom (Residence_type -> residence_type)
df.columns = [c.strip().lower() for c in df.columns]

# c) bmi harus numerik (jika terbaca teks, mis. 'N/A', jadikan NaN)
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

# d) hypertension, heart_disease, stroke tersimpan sebagai angka tetapi
#    sebenarnya kategori biner 0/1 -> jadikan integer kecil (bukan kontinu)
KOLOM_BINER = ["hypertension", "heart_disease"]
for c in KOLOM_BINER + [TARGET]:
    df[c] = df[c].astype("int8")

# e) age di dataset ini bertipe float (ada usia bayi, mis. 0.08) -> tetap float
KOLOM_NUMERIK = ["age", "avg_glucose_level", "bmi"]
KOLOM_NOMINAL = ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]

print("\nTipe kolom setelah koreksi:")
print(df.dtypes.to_string())

# %% [markdown]
# ## 3. Kategori: cek ejaan / kapitalisasi / nilai langka

# %%
garis("3. PEMBERSIHAN KATEGORI")
for c in KOLOM_NOMINAL:
    sebelum = sorted(df[c].dropna().astype(str).unique())
    df[c] = (
        df[c].astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )
    sesudah = sorted(df[c].dropna().unique())
    print(f"\n[{c}]\n  sebelum : {sebelum}\n  sesudah : {sesudah}")

# Kategori sangat langka (<= 5 baris) tidak bisa dipelajari model -> dibuang
for c in KOLOM_NOMINAL:
    vc = df[c].value_counts()
    langka = vc[vc <= 5].index.tolist()
    if langka:
        n = df[c].isin(langka).sum()
        print(f"\nKategori langka pada '{c}': {langka} ({n} baris) -> baris dibuang")
        df = df[~df[c].isin(langka)]

# 'unknown' pada smoking_status dipertahankan sebagai kategori tersendiri
# (proporsinya besar, membuangnya menghilangkan banyak data).
print("\nProporsi smoking_status:")
print((df["smoking_status"].value_counts(normalize=True) * 100).round(1).to_string())

# %% [markdown]
# ## 4. Duplikasi

# %%
garis("4. DUPLIKASI")
n_dup = df.duplicated().sum()
print(f"Jumlah baris duplikat (setelah kolom id dibuang): {n_dup}")
if n_dup:
    df = df.drop_duplicates()
    print("Baris duplikat dibuang.")
df = df.reset_index(drop=True)
print("Ukuran data setelah cek duplikasi:", df.shape)

# %% [markdown]
# ## 5. Missing values (identifikasi)

# %%
garis("5. MISSING VALUES")
mv = df.isna().sum()
mv_pct = (mv / len(df) * 100).round(2)
print(pd.DataFrame({"jumlah_missing": mv, "persen": mv_pct}).to_string())
print("\nCatatan: 'unknown' pada smoking_status diperlakukan sebagai kategori sendiri.")

# %% [markdown]
# ## 6. Pisahkan data latih/uji DULU (agar tidak terjadi data leakage)
# Semua nilai statistik (median, batas IQR, mean/std, kategori) dihitung hanya dari
# data latih, lalu diterapkan ke data uji.

# %%
garis("6. PEMBAGIAN DATA (stratified 80:20)")
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print("X_train:", X_train.shape, "| X_test:", X_test.shape)

# Cek leakage 1: apakah ada baris identik di data latih dan uji?
irisan = X_train.merge(X_test, how="inner").shape[0]
print(f"Baris identik antara data latih & uji: {irisan}")

X_train = X_train.copy()
X_test = X_test.copy()

# %% [markdown]
# ## 7. Missing values: imputasi (median dari data latih)

# %%
garis("7. IMPUTASI MISSING VALUE (bmi)")
median_bmi = X_train["bmi"].median()
print(f"Median bmi (data latih) = {median_bmi:.2f}")
print("Missing bmi sebelum imputasi -> train:", X_train["bmi"].isna().sum(),
      "| test:", X_test["bmi"].isna().sum())
X_train["bmi"] = X_train["bmi"].fillna(median_bmi)
X_test["bmi"] = X_test["bmi"].fillna(median_bmi)
print("Missing bmi setelah imputasi  -> train:", X_train["bmi"].isna().sum(),
      "| test:", X_test["bmi"].isna().sum())
print("Total missing tersisa:", int(X_train.isna().sum().sum() + X_test.isna().sum().sum()))

# %% [markdown]
# ## 8. (Opsional) Outlier dengan metode IQR

# %%
garis("8. OUTLIER (IQR) - dihitung pada data latih")


def batas_iqr(s, k=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


hasil = []
for c in ["bmi", "avg_glucose_level"]:
    lo, hi = batas_iqr(X_train[c])
    n_out = int(((X_train[c] < lo) | (X_train[c] > hi)).sum())
    hasil.append({"kolom": c, "batas_bawah": round(lo, 2), "batas_atas": round(hi, 2),
                  "jml_outlier": n_out, "persen": round(n_out / len(X_train) * 100, 2),
                  "min": X_train[c].min(), "max": X_train[c].max()})
print(pd.DataFrame(hasil).to_string(index=False))

# Keputusan:
#  - bmi: nilai ekstrem (mis. > 60) tidak realistis -> di-cap (winsorizing) ke batas IQR
#  - avg_glucose_level: nilai tinggi = penderita diabetes (valid secara klinis dan
#    informatif untuk stroke) -> dilaporkan saja, TIDAK diubah
lo_bmi, hi_bmi = batas_iqr(X_train["bmi"])
X_train["bmi"] = X_train["bmi"].clip(lo_bmi, hi_bmi)
X_test["bmi"] = X_test["bmi"].clip(lo_bmi, hi_bmi)
print(f"\nbmi di-cap ke rentang [{lo_bmi:.2f}, {hi_bmi:.2f}] (batas dari data latih)")

# %% [markdown]
# ## 9. Normalisasi & standarisasi + encoding kategori

# %%
garis("9. NORMALISASI, STANDARISASI & ENCODING")

# Demo normalisasi Min-Max (hanya untuk perbandingan, fit di data latih)
mm = MinMaxScaler().fit(X_train[KOLOM_NUMERIK])
demo_mm = pd.DataFrame(mm.transform(X_train[KOLOM_NUMERIK]), columns=KOLOM_NUMERIK)
print("Normalisasi Min-Max (rentang 0-1) - min/max data latih setelah transformasi:")
print(demo_mm.agg(["min", "max"]).round(3).to_string())

# Pipeline final: Standardisasi (z-score) untuk numerik, One-Hot untuk nominal,
# kolom biner (hypertension, heart_disease) dibiarkan apa adanya (sudah 0/1).
prep = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), KOLOM_NUMERIK),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                              sparse_output=False), KOLOM_NOMINAL),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
prep.set_output(transform="pandas")

X_train_p = prep.fit_transform(X_train)
X_test_p = prep.transform(X_test)

print("\nMean & std kolom numerik setelah standarisasi (data latih) -> harus ~0 dan ~1:")
print(X_train_p[KOLOM_NUMERIK].agg(["mean", "std"]).round(3).to_string())
print(f"\nJumlah fitur setelah encoding: {X_train_p.shape[1]}")
print("Daftar fitur akhir:", list(X_train_p.columns))

# %% [markdown]
# ## 10. (Opsional) Ketidakseimbangan kelas

# %%
garis("10. KETIDAKSEIMBANGAN KELAS")
dist = y_train.value_counts().rename({0: "tidak stroke (0)", 1: "stroke (1)"})
print("Distribusi target data latih:")
print(pd.DataFrame({"jumlah": dist, "persen": (dist / len(y_train) * 100).round(2)}).to_string())
rasio = y_train.value_counts().max() / y_train.value_counts().min()
print(f"\nRasio mayoritas : minoritas = {rasio:.1f} : 1  -> data sangat tidak seimbang")

# Random oversampling kelas minoritas HANYA pada data latih
tr = X_train_p.assign(**{TARGET: y_train.values})
mayor = tr[tr[TARGET] == 0]
minor = tr[tr[TARGET] == 1]
minor_up = resample(minor, replace=True, n_samples=len(mayor), random_state=RANDOM_STATE)
tr_bal = pd.concat([mayor, minor_up]).sample(frac=1, random_state=RANDOM_STATE)
print("\nDistribusi setelah random oversampling (hanya data latih):")
print(tr_bal[TARGET].value_counts().to_string())
print("\nCatatan: data uji TIDAK di-oversampling. Evaluasi gunakan Precision/Recall/F1/ROC-AUC,"
      " bukan hanya akurasi. Alternatif lain: SMOTE (imbalanced-learn) atau class_weight='balanced'.")

# %% [markdown]
# ## 11. (Opsional) Indikasi kebocoran data (leakage)

# %%
garis("11. INDIKASI LEAKAGE")
korelasi = (
    X_train_p.assign(**{TARGET: y_train.values})
    .corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
)
print("Korelasi absolut fitur terhadap target (5 teratas):")
print(korelasi.head(5).round(3).to_string())
print("\nPemeriksaan:")
print(" - Kolom 'id' sudah dibuang                           : OK")
print(f" - Baris identik train vs test                        : {irisan} baris")
print(" - Imputasi/IQR/scaler/encoder di-fit hanya di train  : OK")
curiga = korelasi[korelasi > 0.9]
print(f" - Fitur dengan korelasi > 0.9 terhadap target        : "
      f"{'TIDAK ADA' if curiga.empty else list(curiga.index)}")

# %% [markdown]
# ## 12. Hasil akhir: 5 teratas & 5 terbawah data hasil pemrosesan

# %%
df_final = pd.concat(
    [X_train_p.assign(**{TARGET: y_train.values}),
     X_test_p.assign(**{TARGET: y_test.values})]
).sort_index()

tampil_atas_bawah(df_final, "12. DATA HASIL PEMROSESAN (gabungan latih + uji, urut indeks asli)")
tampil_atas_bawah(X_train_p.assign(**{TARGET: y_train.values}), "DATA LATIH HASIL PEMROSESAN")
tampil_atas_bawah(X_test_p.assign(**{TARGET: y_test.values}), "DATA UJI HASIL PEMROSESAN")

df_final.to_csv("stroke_preprocessed.csv", index=False)
print("\nFile hasil disimpan: stroke_preprocessed.csv")
