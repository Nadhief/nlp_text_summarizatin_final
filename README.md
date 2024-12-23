# Text Summarization using Simple RNN, LSTM, Transformers, MT-5-small, dan phi 3.5

Proyek ini adalah implementasi sederhana untuk membuat ringkasan dari sebuah artikel secara otomatis menggunakan model Recurrent Neural Network (RNN), Long Short Term Memory (LSTM), Transformers, MT-5-small, dan phi 3.5. Model dilatih pada data berita untuk membuat ringkasan dari teks artikel. Proyek ini dijalankan di local dan google colab. lalu dataset yang tigunakan untuk training dan finetuned memanfaatkan dataset dari huging face https://huggingface.co/datasets/SEACrowd/liputan6.

# Power Poin Hasil Eksplorasi

https://www.canva.com/design/DAGWVxH4qgk/PqXKA0y2swssjpfOlU_q9g/edit?utm_content=DAGWVxH4qgk&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

# Kode Program yang digunakan

- summaryindoRnn_final (500 data train, 100 data val)
- summaryindoRnn_final_10000 (10000 data train, 2000 data val)
- summaryindolstm_final (500 data train, 100 data val)
- summaryindolstm_final_10000 (10000 data train, 2000 data val)
- summaryindotrans_final (500 data train, 100 data val)
- summaryindotrans_final_1000 (1000 data train, 200 data val)
- mt5_finetuned_100 (100 data train, 20 data val)
- mt5_finetuned_1000 (1000 data train, 200 data val)
- mt5_finetuned_10000 (10000 data train, 2000 data val)
- finetunedMT_5_final 
- phi3.5_final (one shot dan few shot)
- finetuned_phi3.5_final_oneshot_fewshot (100 data train, 20 data val)
  
## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Creator](#project-creator)


## Overview

Ringkasan teks otomatis (text summarization) adalah teknik Natural Language Processing (NLP) untuk membuat versi pendek dari suatu teks sambil mempertahankan makna utamanya. Proyek ini menggunakan **RNN**, **LSTM**, **TRANSFORMERS**, **MT-5-SMALL**, dan **PHI3.5** untuk menghasilkan ringkasan dari artikel berita.

## Features

*RNN:*
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

*MT-5:*

- **Data Preprocessing**:  
  - Gunakan tokenizer berbasis model MT-5 untuk mengonversi teks menjadi token numerik dengan *padding* agar semua token memiliki panjang yang seragam.  
  - Pisahkan data menjadi *training set*, *validation set*, dan *test set* untuk proses pelatihan dan evaluasi.  
  - Pastikan setiap input-output pasangan teks disesuaikan dengan format *seq2seq* yang diharapkan oleh model MT-5.

- **Model MT-5**:  
  - Model *Multilingual T5 (MT-5)* adalah *transformer encoder-decoder* yang mendukung banyak bahasa, digunakan untuk tugas-tugas *text-to-text*.  
  - Model dilatih untuk menghasilkan teks keluaran berdasarkan teks masukan, memanfaatkan kemampuannya dalam pemrosesan bahasa alami multibahasa.

- **Training and Evaluation**:  
  - Latih model menggunakan data *training set*, dengan memonitor *Loss* pada *validation set* untuk memastikan stabilitas pelatihan.  
  - Gunakan optimasi *AdamW* dan *learning rate scheduling* untuk meningkatkan efisiensi pelatihan.  
  - Visualisasikan metrik seperti *Loss* selama proses pelatihan.

- **Prediction Function**:  
  - Buat fungsi untuk menghasilkan prediksi ringkasan menggunakan tokenizer dan model MT-5 terlatih.  
  - Pastikan teks input diproses dengan tokenizer MT-5 sebelum dimasukkan ke model untuk prediksi.  

- **Evaluation with BLEU Score**:  
  - Gunakan *BLEU score* untuk mengevaluasi kualitas ringkasan yang dihasilkan pada *test set*.  
  - Hitung skor BLEU untuk setiap sampel dan rata-ratakan untuk mendapatkan metrik keseluruhan.

*Phi-3.5-mini-instruct:*

- **Data Preprocessing**:  
  - Gunakan tokenizer bawaan dari model Phi-3.5 untuk mengonversi teks menjadi token numerik. Terapkan *padding* agar semua token memiliki panjang yang seragam.  
  - Pisahkan dataset menjadi *training set* dan *validation set*. Gunakan format input-output yang sesuai dengan spesifikasi *instruction-tuned models*.

- **Model Phi-3.5-mini-instruct**:  
  - Model Phi-3.5-mini-instruct adalah LLM (*Large Language Model*) berukuran lebih kecil yang dirancang untuk tugas-tugas *instruction-following* menggunakan pendekatan *few-shot* atau *one-shot learning*.  
  - Model dilatih untuk memahami instruksi dan menghasilkan keluaran yang relevan dengan teks masukan.

- **Training and Evaluation**:  
  - Fine-tune model dengan data *training set* menggunakan *learning rate* yang disesuaikan untuk mencegah *overfitting*.  
  - Monitor metrik seperti *Loss* dan *Accuracy* pada *validation set* untuk mengevaluasi kinerja model selama pelatihan.  
  - Simpan *checkpoint* terbaik berdasarkan *validation loss* untuk memastikan model yang optimal.

- **Prediction Function**:  
  - Kembangkan fungsi prediksi yang memanfaatkan model Phi-3.5-mini-instruct untuk menghasilkan ringkasan.  
  - Input teks harus di-tokenisasi terlebih dahulu, dan keluaran model harus diterjemahkan kembali ke teks asli untuk evaluasi.

- **Evaluation with BLEU Score**:  
  - Uji model dengan *test set* untuk mengevaluasi kualitas ringkasan menggunakan metrik *BLEU score*.  
  - Hitung dan bandingkan rata-rata skor BLEU untuk memvalidasi hasil prediksi terhadap data referensi.

# Dataset dan Model

Dataset yang digunakan berasal dari https://huggingface.co/datasets/SEACrowd/liputan6 yang berisikan 3 kolom yaitu document, id, summary berbahasa indonesia.
- data train 193883 row
- data test 10972 row
- data validation 10972 row

Model MT-5-small berasa dari https://huggingface.co/google/mt5-small
Model phi3.5-mini-instruct berasa dari https://huggingface.co/microsoft/Phi-3.5-mini-instruct

Contoh data:
- Sample document: TIGA kali somasi dilayangkan kepada Nuri Shaden . Namun reaksi yang diharapkan agar meminta maaf hasilnya nihil . Keluarga Janu Utomo melaporkan kasus kecelakaan di Jalan Sisingamaraja , Jakarta Selatan , 1 Juni silam itu ke Kepolisian Resor Metro Jaksel . " Kita melaporkan atas dugaan tindak pidana , " tutur Taufik Basari , kuasa hukum keluarga Janu , belum lama ini . Taufik menambahkan , pihaknya ingin Nuri mencabut pernyataan di depan pers , 2 Juni silam . Ia juga ingin Nuri meminta maaf di media atas kejadian itu . Dan juga menyampaikan belasungkawa . Taufik menilai , pernyataan Nuri memutarbalikkan fakta . " Yang kita adukan soal pernyataan-pernyataannya , " ujar dia . Keluarga Janu melaporkan dengan dua tuduhan . Yakni sikap yang tidak menyenangkan dan pencemaran nama baik . Kala itu Nuri menuding mobil ambulans yang menabrak kendaraannya . " Intinya mereka menabrak kita , " ucap Nuri saat jumpa pers waktu itu . Pernyataan disampaikan bintang film Seventeen didampingi Noni T , pengacaranya . Ia menyalahkan Januari Purwoko , sopir ambulans . Padahal Januari sudah mengklakson mobil Honda Jazz yang ditumpangi Nuri . Tabrakan pun terhindarkan . Ambulans rusak di bagian kanan dan as roda belakang patah . Tabrakan mobil ambulans dan Honda Jazz yang dikendarai Nuri merenggut nyawa Janu Utomo . Sedangkan Januari Purwoko , sopir ambulans cedera parah . Tiga lainnya di mobil ambulans luka ringan . Yakni istri Janu , Retno Indarti , Krisanti Indriani , dan perawat Risa Citra Dewi . Kala itu keluarga mengantarkan Janu hendak mencuci darah . Soal ini kubu Nuri membantah . Noni mengatakan Janu meninggal karena penyakit jantung yang dideritanya . ( As ) .
- Sample Summary: TIGA kali somasi dilayangkan kepada Nuri Shaden . Namun reaksi yang diharapkan agar meminta maaf hasilnya nihil . Keluarga Janu Utomo pun melaporkan Nuri ke Kepolisian Resor Metro Jakarta Selatan .

## Installation
pip install tensorflow
pip install datasets
pip install nltk
pip install matplotlib
pip install pandas numpy 
pip install transformers
pip install peft

## Project Creator
Proyek ini dikembangkan oleh **Nadhief Athallah Isya** dengan NIM **2106413** kelas **Ilmu Komputer C2 2021**, seorang mahasiswa Ilmu Komputer C2 2021 Universitas Pendidikan Indoensia yang sedang memenuhi tugas **Pengolahan Bahasa Alami**.
