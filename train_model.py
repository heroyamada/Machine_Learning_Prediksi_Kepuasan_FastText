# train_model.py (VERSI TERBARU DENGAN FASTTEXT EMBEDDINGS)

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import gensim.models # Untuk FastText

warnings.filterwarnings('ignore')

print("Memulai proses pelatihan model (versi dengan FastText Embeddings)...\n")

# --- 0. Konfigurasi FastText ---
FASTTEXT_MODEL_PATH = 'cc.id.300.vec' # Ganti dengan path ke model FastText Indonesia Anda (.vec)
# Jika Anda mengunduh cc.id.300.bin, ganti baris di bawah ini:
# fasttext_model = gensim.models.fasttext.load_facebook_model(FASTTEXT_MODEL_PATH)

# --- Pastikan NLTK resources sudah terunduh ---
print("Mengecek dan mendownload NLTK resources (stopwords, punkt, wordnet, omw-1.4)...\n")
try:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("NLTK resources selesai dicek/diunduh.\n")
except Exception as e:
    print(f"Error during NLTK download: {e}")
    print("Pastikan koneksi internet Anda stabil dan coba lagi.")
    exit()


# --- 1. Memuat Dataset ---
try:
    df = pd.read_csv('Data ulasan Shopee tentang COD.csv')
    print("Dataset berhasil dimuat.")
    print(f"Bentuk dataset: {df.shape}")
    print("5 baris pertama dataset:\n", df.head())
    print("\nInformasi dataset:")
    df.info()
    print("\nNama-nama kolom dalam dataset:")
    print(df.columns)
except FileNotFoundError:
    print("Error: 'Data ulasan Shopee tentang COD.csv' tidak ditemukan.")
    print("Pastikan file tersebut berada di direktori yang sama dengan script Anda.")
    exit()

# --- 2. Preprocessing Teks ---
print("\nMemulai preprocessing teks...")
stopwords_id = set(stopwords.words('indonesian'))
if 'lama' in stopwords_id: # Pastikan 'lama' tidak difilter
    stopwords_id.remove('lama')
print(f"Stopwords yang digunakan (setelah menghapus 'lama' jika ada): {list(stopwords_id)[:10]} ... (dan {len(stopwords_id) - 10} lainnya)\n")

lemmatizer = WordNetLemmatizer()

def preprocess_text_full(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stopwords_id]
    lemmas = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(lemmas)

df.dropna(subset=['content'], inplace=True)
df['Content_Clean'] = df['content'].apply(preprocess_text_full)
df = df[df['Content_Clean'] != '']
print("Preprocessing teks selesai.\n")
print("Contoh ulasan setelah preprocessing:\n", df[['content', 'Content_Clean']].head())


# --- 3. Penggabungan dan Encoding Label Target (2 Kelas) ---
print("\nMelakukan penggabungan kelas menjadi 2 kategori (Negatif, Positif) dan encoding label target...\n")
X_text = df['Content_Clean']

def merge_sentiment_classes_2_labels(score):
    if score in [1, 2, 3]: # 'sangat tidak puas', 'tidak puas', 'netral' -> Negatif
        return 'Negatif'
    elif score in [4, 5]: # 'puas', 'sangat puas' -> Positif
        return 'Positif'
    return None

df['Merged_Sentiment'] = df['score'].apply(merge_sentiment_classes_2_labels)
df.dropna(subset=['Merged_Sentiment'], inplace=True)

le = LabelEncoder()
y = le.fit_transform(df['Merged_Sentiment'])
target_names = ['Negatif', 'Positif']

print(f"Label target berhasil di-encode setelah penggabungan menjadi 2 kelas.")
print(f"Kelas Asli Setelah Penggabungan (le.classes_): {list(le.classes_)}")
print(f"Kelas Ter-encode (np.unique(y)): {list(np.unique(y))}")
print(f"Nama kelas untuk laporan dan visualisasi: {target_names}\n")

# --- 4. Memuat Model FastText ---
print("Memuat model FastText Bahasa Indonesia... (Ini mungkin membutuhkan waktu)\n")
try:
    fasttext_model = gensim.models.KeyedVectors.load_word2vec_format(FASTTEXT_MODEL_PATH, binary=False)
    print("Model FastText berhasil dimuat.\n")
except FileNotFoundError:
    print(f"Error: Model FastText tidak ditemukan di '{FASTTEXT_MODEL_PATH}'.")
    print("Pastikan Anda sudah mengunduh dan menempatkan file model FastText (cc.id.300.vec) di lokasi yang benar.")
    exit()
except Exception as e:
    print(f"Error saat memuat model FastText: {e}")
    print("Pastikan format model benar (.vec) dan path sudah sesuai.")
    exit()

# --- 5. Ekstraksi Fitur dengan FastText Embeddings ---
print("Melakukan ekstraksi fitur dengan FastText Embeddings...\n")

def get_sentence_embedding(sentence, model):
    words = sentence.split()
    word_vectors = [model[word] for word in words if word in model] # FIX AttributeError
    if not word_vectors:
        return np.zeros(model.vector_size) 
    return np.mean(word_vectors, axis=0)

X_embeddings = np.array([get_sentence_embedding(text, fasttext_model) for text in X_text])
print(f"Ukuran FastText Embeddings: {X_embeddings.shape}\n")

# --- 6. Pembagian Data (menggunakan X_embeddings) ---
print("Membagi data (menggunakan FastText Embeddings) dan melakukan pelatihan...\n")
X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42, stratify=y)
print(f"Ukuran data pelatihan: {X_train.shape}, Ukuran data pengujian: {X_test.shape}\n")


# --- 7. Pelatihan Model Stacking yang Dioptimalkan ---
print("Membangun dan melatih model Stacking dengan FastText Embeddings...\n")
base_learners = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')),
    ('svc', SVC(probability=True, random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42))
]
meta_learner = LogisticRegression(random_state=42, solver='liblinear')
stacked_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5, n_jobs=-1)

stacked_model.fit(X_train, y_train)
print("Pelatihan model selesai.\n")


# --- 8. Evaluasi Model ---
print("--- HASIL EVALUASI MODEL ---")
y_pred = stacked_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Model Stacking (dengan FastText): {accuracy:.4f}")

print("\nLaporan Klasifikasi Stacked Model:")
print(classification_report(y_test, y_pred, target_names=target_names))

# --- 9. Visualisasi Matriks Kebingungan ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Matriks Kebingungan (Confusion Matrix) - Stacked Model (FastText)')
plt.show()

# --- 10. Kinerja Base Learners (Individu) ---
print("\n--- Kinerja Base Learners (Individu) ---\n")
for name, estimator in base_learners:
    print(f"Evaluasi Model Base: {name.upper()}")
    cloned_estimator = estimator.__class__(**estimator.get_params())
    cloned_estimator.fit(X_train, y_train)

    y_pred_base = cloned_estimator.predict(X_test)

    accuracy_base = accuracy_score(y_test, y_pred_base)
    print(f"Akurasi {name.upper()}: {accuracy_base:.4f}")

    print(f"Laporan Klasifikasi {name.upper()}:")
    print(classification_report(y_test, y_pred_base, target_names=target_names))
    print("-" * 50)


# --- 11. Simpan Model dan LabelEncoder ---
try:
    joblib.dump(stacked_model, 'stacked_model_fasttext.pkl')
    joblib.dump(le, 'label_encoder_fasttext.pkl')
    with open('fasttext_model_path.txt', 'w') as f:
        f.write(FASTTEXT_MODEL_PATH)
    print("\nModel Stacking (FastText), LabelEncoder, dan path model FastText berhasil disimpan.")
except Exception as e:
    print(f"Error saat menyimpan model atau LabelEncoder: {e}")

print("\nProses pelatihan model selesai.")

# --- 12. Test Prediksi dengan Contoh Ulasan ---
print("\n--- MENGUJI MODEL DENGAN CONTOH ULASAN KUSTOM ---")
sample_reviews = [
    "barang cacat",
    "barang bagus",
    "pengiriman sangat cepat dan aman",
    "kecewa berat dengan kualitas produk",
    "produk lumayan tapi pengiriman lama",
    "kurir lama",
    "barang rusak",
    "tidak sesuai",
    "sangat puas sekali"
]

for review in sample_reviews:
    processed_review = preprocess_text_full(review)
    review_embedding = get_sentence_embedding(processed_review, fasttext_model)
    numerical_prediction = stacked_model.predict(review_embedding.reshape(1, -1))[0]
    final_prediction_label = le.inverse_transform([numerical_prediction])[0]
    print(f"Ulasan: '{review}' -> Prediksi: {final_prediction_label}")

print("\nPengujian contoh ulasan selesai.")