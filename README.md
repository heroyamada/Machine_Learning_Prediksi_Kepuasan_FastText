# Machine_Learning_Prediksi_Kepuasan_FastText
link model FastText : https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.vec.gz
letakan pada folder yang sama dengan python app.py dan train_model.py

aplikasi ini wajib dijalankan di python versi 3.11, jika anda mempunyai python versi lebih tinggi maka anda harus menjalankanya di virtual environment, langkahnya seperti ini :

Untuk membuat dan menggunakan folder venv_fasttext_py311 sebagai lingkungan virtual (virtual environment), Anda sebenarnya tidak perlu mengunduh banyak hal terpisah. Yang utama adalah memastikan Python 3.11 sudah terinstal di sistem Anda.

Berikut adalah yang perlu Anda miliki:

1. Python 3.11
Ini adalah syarat paling mendasar. Modul venv yang digunakan untuk membuat lingkungan virtual adalah bagian standar dari instalasi Python.

Bagaimana cara mendapatkannya:

Untuk Windows: Unduh installer dari situs resmi Python: https://www.python.org/downloads/windows/ (cari versi 3.11.x). Pastikan untuk mencentang opsi "Add Python to PATH" saat instalasi.

Untuk macOS: Gunakan Homebrew (rekomendasi): brew install python@3.11. Atau unduh installer dari situs resmi Python: https://www.python.org/downloads/macos/.

Untuk Linux: Gunakan manajer paket sistem Anda (misalnya, sudo apt-get install python3.11 untuk Debian/Ubuntu, atau sudo dnf install python3.11 untuk Fedora).

Cara Memverifikasi: Setelah instalasi, buka terminal atau command prompt dan ketik:

Bash

python3.11 --version
Anda akan melihat output seperti Python 3.11.x. Jika tidak, berarti Python 3.11 belum terinstal atau tidak dikenali di PATH Anda.

2. Pustaka fasttext (Akan Diinstal Setelah Lingkungan Virtual Dibuat)
Pustaka fasttext itu sendiri tidak perlu diunduh sebelum membuat lingkungan virtual. Sebaliknya, Anda akan menginstalnya di dalam venv_fasttext_py311 setelah lingkungan virtual tersebut diaktifkan.
3. Navigasi ke Direktori Proyek Anda:
Ini adalah langkah krusial. Anda harus berada di folder akar proyek Anda di mana semua kode Python Anda berada (dan di mana Anda ingin folder venv_fasttext_py311 ini dibuat).
Gunakan perintah cd (change directory):
Tekan Enter setelah mengetik perintah cd. Pastikan Anda berada di direktori yang benar sebelum melanjutkan.
4. Buat Lingkungan Virtual (venv):
Setelah Anda berada di dalam folder proyek Anda, jalankan perintah ini:

Bash
python3.11 -m venv venv_fasttext_py311
python3.11: Memanggil interpreter Python versi 3.11.
-m venv: Memberi tahu Python untuk menjalankan modul venv (yang khusus untuk membuat lingkungan virtual).
venv_fasttext_py311: Ini adalah nama folder yang akan dibuat untuk lingkungan virtual Anda. Anda bisa memberinya nama apa pun, tetapi venv_fasttext_py311 jelas menunjukkan tujuannya.
Tekan Enter. Proses ini mungkin memakan waktu beberapa detik. Tidak akan ada banyak output di terminal, tetapi Anda akan melihat kursor kembali.

Setelah folder venv_fasttext_py311 muncul, Anda perlu melakukan dua hal penting:
Aktifkan Lingkungan Virtual:
Sebelum Anda menginstal paket (seperti fasttext) atau menjalankan skrip Python Anda di lingkungan ini, Anda harus mengaktifkannya.
Windows (Command Prompt): .venv_fasttext_py311\Scripts\activate

Windows (PowerShell): .\venv_fasttext_py311\Scripts\Activate.ps1

macOS/Linux: source venv_fasttext_py311/bin/activate

Setelah diaktifkan, Anda akan melihat (venv_fasttext_py311) di awal baris perintah terminal Anda.

Anda cukup menjalankan di command prompt:

Bash

pip install fasttext (dan yang lainya)
Perintah pip ini akan mengunduh dan menginstal pustaka fasttext serta semua dependensinya ke dalam lingkungan virtual Anda.

Singkatnya:

Pastikan Anda memiliki Python 3.11 terinstal di komputer Anda. Itu saja yang Anda butuhkan untuk membuat folder venv_fasttext_py311. Setelah itu, Anda akan menggunakan pip (yang sudah disertakan dengan instalasi Python) untuk mengunduh dan menginstal pustaka fasttext ke dalam lingkungan virtual tersebut.
