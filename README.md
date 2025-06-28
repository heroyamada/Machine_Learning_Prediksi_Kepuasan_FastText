# Machine_Learning_Prediksi_Kepuasan_FastText
link model FastText : https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.vec.gz

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

Bagaimana cara mendapatkannya:
Setelah Anda mengaktifkan lingkungan virtual (source venv_fasttext_py311/bin/activate di Linux/macOS atau .venv_fasttext_py311\Scripts\activate di Windows), Anda cukup menjalankan:

Bash

pip install fasttext
Perintah pip ini akan mengunduh dan menginstal pustaka fasttext serta semua dependensinya ke dalam lingkungan virtual Anda.

Singkatnya:

Pastikan Anda memiliki Python 3.11 terinstal di komputer Anda. Itu saja yang Anda butuhkan untuk membuat folder venv_fasttext_py311. Setelah itu, Anda akan menggunakan pip (yang sudah disertakan dengan instalasi Python) untuk mengunduh dan menginstal pustaka fasttext ke dalam lingkungan virtual tersebut.
