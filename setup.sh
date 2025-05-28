#!/bin/bash

# Pastikan direktori NLTK data ada
mkdir -p /home/appuser/.nltk_data/

# Download NLTK resources ke direktori yang ditentukan
python -m nltk.downloader -d /home/appuser/.nltk_data stopwords
python -m nltk.downloader -d /home/appuser/.nltk_data wordnet
python -m nltk.downloader -d /home/appuser/.nltk_data omw-1.4
python -m nltk.downloader -d /home/appuser/.nltk_data punkt # Tambahkan ini juga, sering dibutuhkan

# (Opsional) Verifikasi bahwa direktori ada setelah download
# ls -l /home/appuser/.nltk_data/corpora/
# ls -l /home/appuser/.nltk_data/tokenizers/