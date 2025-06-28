# app.py (VERSI FINAL DENGAN FASTTEXT EMBEDDINGS & PERBAIKAN DB)
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, get_flashed_messages
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import mysql.connector
from math import ceil
import numpy as np
import gensim.models

# Inisialisasi aplikasi Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'kunci_rahasia_super_aman_anda_di_sini_12345' # GANTI DENGAN KUNCI ACAK YANG KUAT!

# --- Konfigurasi Database MySQL ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'dbulasan'
}

# --- Konstanta Paginasi ---
REVIEWS_PER_PAGE = 4

# --- Muat Model, Vectorizer, dan LabelEncoder ---
stacked_model = None
label_encoder = None
fasttext_model = None
stopwords_id = set(stopwords.words('indonesian'))
lemmatizer = WordNetLemmatizer()

# --- Pastikan NLTK resources sudah terunduh (untuk aplikasi web) ---
print("Mengecek dan mendownload NLTK resources untuk aplikasi web (stopwords, punkt, wordnet, omw-1.4)...\n")
try:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("NLTK resources selesai dicek/diunduh.\n")
except Exception as e:
    print(f"Error during NLTK download in app.py: {e}")
    print("Pastikan koneksi internet Anda stabil dan coba lagi.")
    exit()

# --- Pastikan 'lama' tidak dihilangkan dari stopwords ---
if 'lama' in stopwords_id:
    stopwords_id.remove('lama')
print(f"Stopwords yang digunakan di app.py (setelah menghapus 'lama' jika ada): {list(stopwords_id)[:10]} ... (dan {len(stopwords_id) - 10} lainnya)\n")

try:
    # Muat path model FastText dari file
    with open('fasttext_model_path.txt', 'r') as f:
        fasttext_model_path = f.read().strip()

    # Muat model FastText (asumsi .vec, sesuaikan jika .bin)
    fasttext_model = gensim.models.KeyedVectors.load_word2vec_format(fasttext_model_path, binary=False)
    
    stacked_model = joblib.load('stacked_model_fasttext.pkl')
    label_encoder = joblib.load('label_encoder_fasttext.pkl')
    print("Model Stacking, FastText Model, dan LabelEncoder berhasil dimuat.\n")
except FileNotFoundError as e:
    print(f"Error: Salah satu file model tidak ditemukan: {e}.")
    print("Pastikan Anda sudah menjalankan script train_model.py terlebih dahulu untuk melatih dan menyimpan model FastText.")
    exit()
except Exception as e:
    print(f"Error saat memuat model atau LabelEncoder di app.py: {e}")
    print("Pastikan file model FastText ada di lokasi yang benar dan formatnya sesuai.")
    exit()

# --- Fungsi Preprocessing Teks (Disatukan dengan train_model.py's preprocess_text_full) ---
def preprocess_text_for_prediction(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stopwords_id]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return ' '.join(lemmatized_tokens)

# --- Fungsi untuk mendapatkan FastText Embedding (baru di app.py) ---
def get_sentence_embedding_for_app(sentence, model):
    words = sentence.split()
    word_vectors = [model[word] for word in words if word in model] # Akses langsung dari model KeyedVectors
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)


# --- Fungsi Database ---
def get_db_connection():
    print("DEBUG DB: Mencoba koneksi ke database...")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        print("DEBUG DB: Koneksi database berhasil.")
        return conn
    except mysql.connector.Error as err:
        print(f"DEBUG DB ERROR: Gagal koneksi ke database: {err}")
        return None

def save_review_prediction(review_text, prediction):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            query = "INSERT INTO reviews_history (review_text, prediction) VALUES (%s, %s)"
            cursor.execute(query, (review_text, prediction))
            conn.commit()
            print(f"DEBUG DB: Review saved to DB: '{review_text}' -> '{prediction}'")
        except mysql.connector.Error as err:
            print(f"DEBUG DB ERROR: Gagal menyimpan review ke database: {err}")
        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if conn:
                conn.close()

def get_paginated_reviews(page, per_page):
    conn = get_db_connection()
    reviews = []
    total_reviews = 0
    if conn:
        try:
            cursor = conn.cursor(dictionary=True) # Kursor mengembalikan dictionary
            print(f"DEBUG DB: Mengambil total ulasan untuk paginasi...")
            count_query = "SELECT COUNT(*) AS total_count FROM reviews_history" # Tambahkan alias untuk lebih jelas
            cursor.execute(count_query)
            
            # PERBAIKAN UTAMA: Akses hasil COUNT(*) dengan kunci dictionary
            count_result = cursor.fetchone()
            if count_result and 'total_count' in count_result:
                total_reviews = count_result['total_count']
            else:
                total_reviews = 0 # Jika tidak ada hasil, anggap 0

            print(f"DEBUG DB: Total ulasan ditemukan: {total_reviews}")

            offset = (page - 1) * per_page
            if offset < 0:
                offset = 0

            print(f"DEBUG DB: Mengambil ulasan untuk halaman {page}, offset {offset}, limit {per_page}...")
            data_query = "SELECT id, review_text, prediction, timestamp FROM reviews_history ORDER BY id ASC LIMIT %s OFFSET %s"
            cursor.execute(data_query, (per_page, offset))
            reviews = cursor.fetchall()
            print(f"DEBUG DB: Berhasil mengambil {len(reviews)} ulasan untuk halaman {page}.")
        except mysql.connector.Error as err:
            print(f"DEBUG DB ERROR: Gagal mengambil ulasan dari database: {err}")
        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if conn:
                conn.close()
    return reviews, total_reviews

@app.route('/')
def index():
    print("DEBUG: Index route diakses.")
    page = request.args.get('page', 1, type=int)
    reviews_history = []
    total_reviews = 0
    total_pages = 1

    try:
        reviews_history, total_reviews = get_paginated_reviews(page, REVIEWS_PER_PAGE)
        total_pages = ceil(total_reviews / REVIEWS_PER_PAGE) if total_reviews > 0 else 1

        # Logika pengalihan halaman setelah penghapusan/penambahan
        # Pastikan kita tidak melebihi halaman terakhir yang valid
        if page > total_pages and total_pages > 0:
            page = total_pages # Kembali ke halaman terakhir yang valid
            reviews_history, total_reviews = get_paginated_reviews(page, REVIEWS_PER_PAGE) # Ambil ulang data untuk halaman yang benar
        elif total_pages == 0:
            page = 1
            reviews_history = [] # Pastikan kosong jika tidak ada ulasan
        
        print(f"DEBUG: Data siap dirender. reviews_history count: {len(reviews_history)}, current_page: {page}, total_pages: {total_pages}")
    except Exception as e:
        print(f"DEBUG ERROR: Kesalahan saat memproses data untuk rendering halaman indeks: {e}")
        # Jika ada error, set data kosong agar halaman tidak crash dan tampilkan flash message
        reviews_history = []
        total_reviews = 0
        total_pages = 1
        flash(f"Terjadi kesalahan fatal saat memuat riwayat ulasan: {e}. Mohon periksa konfigurasi database Anda.", 'error')

    return render_template('index.html',
                            reviews_history=reviews_history,
                            current_page=page,
                            total_pages=total_pages)

@app.route('/predict', methods=['POST'])
def predict():
    review_text = request.form['review_text']
    if not review_text:
        flash("Mohon masukkan ulasan untuk diprediksi.", 'error')
        return redirect(url_for('index', page=request.args.get('page', 1, type=int)))

    processed_review = preprocess_text_for_prediction(review_text)
    
    review_embedding = get_sentence_embedding_for_app(processed_review, fasttext_model)
    numerical_prediction = stacked_model.predict(review_embedding.reshape(1, -1))[0]
    
    final_prediction = label_encoder.inverse_transform([numerical_prediction])[0]

    print(f"DEBUG: Ulasan '{review_text}'. Prediksi model numerik: {numerical_prediction}, Prediksi akhir: '{final_prediction}'.")

    save_review_prediction(review_text, final_prediction)

    # Logika untuk mengarahkan ke halaman terakhir setelah penambahan ulasan
    conn = get_db_connection()
    new_total_reviews = 0
    if conn:
        try:
            cursor = conn.cursor() # Ini bukan dictionary=True, jadi [0] itu benar
            count_query = "SELECT COUNT(*) FROM reviews_history"
            cursor.execute(count_query)
            new_total_reviews = cursor.fetchone()[0]
        except mysql.connector.Error as err:
            print(f"Error getting total reviews after prediction: {err}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    new_total_pages = ceil(new_total_reviews / REVIEWS_PER_PAGE) if new_total_reviews > 0 else 1

    prediction_label_text_class = ''
    if final_prediction == 'Positif':
        prediction_label_text_class = 'positif'
    elif final_prediction == 'Negatif':
        prediction_label_text_class = 'negatif'
    else:
        prediction_label_text_class = 'netral' # Fallback, should not be reached

    prediction_html = f"""
    <p>Ulasan :</p>
    <p><em>"{review_text}"</em></p>
    <p>Hasil Prediksi Kepuasan:</p>
    <p class="prediction-label {prediction_label_text_class}">
        <strong>{final_prediction}</strong>
    </p>
    """
    flash_category = final_prediction.lower().replace(" ", "-")
    flash(prediction_html, flash_category) 

    return redirect(url_for('index', page=new_total_pages))

@app.route('/delete_review/<int:review_id>', methods=['POST'])
def delete_review(review_id):
    conn = get_db_connection()
    cursor = None
    count_cursor = None
    if conn:
        try:
            cursor = conn.cursor()
            query = "DELETE FROM reviews_history WHERE id = %s"
            cursor.execute(query, (review_id,))
            conn.commit()
            print(f"DEBUG: Review ID {review_id} deleted from DB.")

            current_page = request.args.get('current_page', 1, type=int)

            count_cursor = conn.cursor()
            count_query = "SELECT COUNT(*) FROM reviews_history"
            count_cursor.execute(count_query)
            new_total_reviews = count_cursor.fetchone()[0]
            
            new_total_pages = ceil(new_total_reviews / REVIEWS_PER_PAGE) if new_total_reviews > 0 else 1

            redirect_page = current_page
            if current_page > new_total_pages:
                redirect_page = new_total_pages
            if redirect_page < 1:
                redirect_page = 1
            
            return redirect(url_for('index', page=redirect_page))

        except mysql.connector.Error as err:
            print(f"Error deleting review from database: {err}")
            current_page = request.args.get('current_page', 1, type=int)
            return redirect(url_for('index', page=current_page))
        finally:
            if cursor:
                cursor.close()
            if count_cursor:
                count_cursor.close()
            if conn:
                conn.close()
    
    current_page = request.args.get('current_page', 1, type=int)
    return redirect(url_for('index', page=current_page))

# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)
it