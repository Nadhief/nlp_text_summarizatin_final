# Text Summarization using Simple RNN, LSTM, dan Transformers

Proyek ini adalah implementasi sederhana untuk membuat ringkasan dari sebuah artikel secara otomatis menggunakan model Recurrent Neural Network (RNN) dan Long Short Term Memory (LSTM). Model dilatih pada data berita untuk membuat ringkasan dari teks artikel. Proyek ini dijalankan di local dan memanfaatkan dataset dari huging face https://huggingface.co/datasets/SEACrowd/liputan6. 

# Power Poin Hasil Eksplorasi

https://www.canva.com/design/DAGWVxH4qgk/PqXKA0y2swssjpfOlU_q9g/edit?utm_content=DAGWVxH4qgk&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

# Kode Program yang digunakan

- summaryindo.ipynb (500 data train, 100 data val, 10.000 vocab_size)
- summaryindolstm.ipynb (500 data train, 100 data val, 10.000 vocab_size)
- summaryindoRnnSd.ipynb (10.000 data train, 2.000 data val, 20.000 vocab_size)
- summaryindolstm copy.ipynb (10.000 data train, 2.000 data val, 20.000 vocab_size)
- summaryindotrans.ipynb (500 data train, 100 data val, 30522 vocab_size)
- summaryindotrans copy (10000 data train, 2000 data val, 30522 vocab_size)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Creator](#project-creator)


## Overview

Ringkasan teks otomatis (text summarization) adalah teknik Natural Language Processing (NLP) untuk membuat versi pendek dari suatu teks sambil mempertahankan makna utamanya. Proyek ini menggunakan **Simple RNN** dan **LSTM** untuk menghasilkan ringkasan dari artikel berita.

## Features

*Simple RNN:*
- **Data Preprocessing**:
  - Mengubah teks menjadi *sequences* numerik menggunakan tokenizer, dan lakukan *padding* agar semua *sequences* memiliki panjang yang sama.
  - Pisahkan data menjadi *training set* dan *validation set* untuk evaluasi kinerja model selama proses pelatihan.

- **Model Simple RNN**:
  - Membangun model dengan layer *Embedding* sebagai representasi teks numerik.
  - Tambahkan layer *Simple RNN* untuk menangkap pola dasar dalam data.
  - Tambahkan layer *Dense* di bagian akhir untuk memproses keluaran dari *Simple RNN* menjadi ringkasan yang diinginkan.
  
- **Training and Evaluation**:
  - Latih model menggunakan *training set* dan pantau metrik *Loss* dan *Accuracy* pada *validation set*.
  - Visualisasikan metrik *Loss* dan *Accuracy* untuk memeriksa stabilitas dan efektivitas pelatihan model.

- **Prediction Function**:
  - Buat fungsi prediksi untuk menghasilkan ringkasan otomatis dari input teks. Fungsi ini menggunakan preprocessing yang sama pada teks input, lalu memanfaatkan model *Simple RNN* terlatih untuk menghasilkan prediksi ringkasan.

- **Evaluation with BLEU Score**:
  - Setelah pelatihan selesai, evaluasi model dengan *BLEU score* menggunakan *validation set* untuk mengukur kualitas ringkasan.

*LSTM:*
- **Data Preprocessing**: 
  - Mengonversi teks menjadi *sequences* numerik menggunakan tokenizer, dan menerapkan *padding* agar semua *sequences* memiliki panjang yang sama. 
  - Pisahkan data menjadi *training* dan *validation set* untuk proses evaluasi selama pelatihan.

- **Model LSTM**:
  - Membangun model dengan layer *Embedding* diikuti oleh layer *LSTM* untuk menangkap informasi konteks dalam jangka panjang.
  - Tambahkan layer *Dense* di bagian akhir untuk memproses keluaran dari LSTM menjadi ringkasan yang diinginkan.
  
- **Training and Evaluation**:
  - Melatih model menggunakan data *training set* dan memantau *Loss* serta *Accuracy* pada *validation set*.
  - Visualisasikan *Loss* dan *Accuracy* untuk memastikan model mengalami proses pelatihan yang stabil.

- **Prediction Function**:
  - Membuat fungsi prediksi yang akan menghasilkan ringkasan dari input teks pengguna. Proses ini mencakup langkah preprocessing yang sama, kemudian model LSTM terlatih akan digunakan untuk menghasilkan ringkasan.

- **Evaluation with BLEU Score**:
  - Setelah melatih model, evaluasi kinerja model dengan *BLEU score* menggunakan *validation set*.


*Transformers:*
- **Data Preprocessing**: 
  - Mengonversi teks menjadi *sequences* numerik menggunakan tokenizer, dan menerapkan *padding* agar semua *sequences* memiliki panjang yang sama. 
  - Pisahkan data menjadi *training* dan *validation set* untuk proses evaluasi selama pelatihan.

- **Model Transformers**:
  
- **Training and Evaluation**:
  - Melatih model menggunakan data *training set* dan memantau *Loss* serta *Accuracy* pada *validation set*.
  - Visualisasikan *Loss* dan *Accuracy* untuk memastikan model mengalami proses pelatihan yang stabil.

- **Prediction Function**:
  - Membuat fungsi prediksi yang akan menghasilkan ringkasan dari input teks pengguna. Proses ini mencakup langkah preprocessing yang sama, kemudian model Transformers terlatih akan digunakan untuk menghasilkan ringkasan.

- **Evaluation with BLEU Score**:
  - Setelah melatih model, evaluasi kinerja model dengan *BLEU score* menggunakan *validation set*.

# Dataset

Dataset yang digunakan berasal dari https://huggingface.co/datasets/SEACrowd/liputan6 yang berisikan 3 kolom yaitu document, id, summary berbahasa indonesia.
- data train 193883 row
- data test 10972 row
- data validation 10972 row

Contoh data:
- Sample document: TIGA kali somasi dilayangkan kepada Nuri Shaden . Namun reaksi yang diharapkan agar meminta maaf hasilnya nihil . Keluarga Janu Utomo melaporkan kasus kecelakaan di Jalan Sisingamaraja , Jakarta Selatan , 1 Juni silam itu ke Kepolisian Resor Metro Jaksel . " Kita melaporkan atas dugaan tindak pidana , " tutur Taufik Basari , kuasa hukum keluarga Janu , belum lama ini . Taufik menambahkan , pihaknya ingin Nuri mencabut pernyataan di depan pers , 2 Juni silam . Ia juga ingin Nuri meminta maaf di media atas kejadian itu . Dan juga menyampaikan belasungkawa . Taufik menilai , pernyataan Nuri memutarbalikkan fakta . " Yang kita adukan soal pernyataan-pernyataannya , " ujar dia . Keluarga Janu melaporkan dengan dua tuduhan . Yakni sikap yang tidak menyenangkan dan pencemaran nama baik . Kala itu Nuri menuding mobil ambulans yang menabrak kendaraannya . " Intinya mereka menabrak kita , " ucap Nuri saat jumpa pers waktu itu . Pernyataan disampaikan bintang film Seventeen didampingi Noni T , pengacaranya . Ia menyalahkan Januari Purwoko , sopir ambulans . Padahal Januari sudah mengklakson mobil Honda Jazz yang ditumpangi Nuri . Tabrakan pun terhindarkan . Ambulans rusak di bagian kanan dan as roda belakang patah . Tabrakan mobil ambulans dan Honda Jazz yang dikendarai Nuri merenggut nyawa Janu Utomo . Sedangkan Januari Purwoko , sopir ambulans cedera parah . Tiga lainnya di mobil ambulans luka ringan . Yakni istri Janu , Retno Indarti , Krisanti Indriani , dan perawat Risa Citra Dewi . Kala itu keluarga mengantarkan Janu hendak mencuci darah . Soal ini kubu Nuri membantah . Noni mengatakan Janu meninggal karena penyakit jantung yang dideritanya . ( As ) .
- Sample Summary: TIGA kali somasi dilayangkan kepada Nuri Shaden . Namun reaksi yang diharapkan agar meminta maaf hasilnya nihil . Keluarga Janu Utomo pun melaporkan Nuri ke Kepolisian Resor Metro Jakarta Selatan .

## Installation
pip install tensorflow
pip install datasets
pip install nltk
pip install matplotlib
pip install pandas numpy 


## Project Creator
Proyek ini dikembangkan oleh **Nadhief Athallah Isya** dengan NIM **2106413** kelas **Ilmu Komputer C2 2021**, seorang mahasiswa Ilmu Komputer C2 2021 Universitas Pendidikan Indoensia yang sedang memenuhi tugas **Pengolahan Bahasa Alami**.
