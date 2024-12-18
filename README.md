# ✨ FishDetect: Klasifikasi Otomatis Jenis Ternak Ikan dengan CNN dan MobileNetV2 ✨

## Overview Project
Proyek ini bertujuan untuk mengembangkan sebuah sistem klasifikasi gambar yang dapat mengenali dan membedakan gambar-gambar dari berbagai jenis ikan ternak menggunakan teknik Convolutional Neural Networks (CNN) dan MobileNetV2. Sistem ini dapat diterapkan untuk otomatisasi dalam industri perikanan atau sebagai komponen aplikasi yang membutuhkan kemampuan klasifikasi gambar ikan secara cepat dan akurat.

**Link Dataset yang digunakan:** [A Large Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset/data). 
Preprocessing yang digunakan antara lain Resizing, Normalization.

Model yang digunakan: ***Convolutional Neural Network*** (CNN) dengan 3 Layer dan Pre Trained Model ***MobileNetV2*** dengan Architecture Model kurang lebih seperti gambar berikut.

**CNN Architecture** ✨

![image](assets/1_CNNarch.jpeg)

**MobileNetV2 Architecture** ✨

![image](assets/2_mobilenetv2arch.png)

## Overview Dataset
Dataset yang digunakan adalah RPS Dataset dengan link sebagai [berikut](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset/data).Dataset ini awalnya terdiri atas 18.000 gambar ikan (9.000 gambar RGB dan 9.000 gambar ground truth), namun setelah dilakukan proses seleksi dan pengaturan, jumlah data yang digunakan untuk pelatihan telah dibatasi menjadi 10.000 gambar. Dataset ini telah dibagi dengan rapi dan disesuaikan untuk keperluan klasifikasi ikan dengan berbagai kategori, sehingga dapat digunakan secara efektif dalam pengujian dan pelatihan model.  data yang terbagi menjadi 70% sebagai *Training Set*, 20% sebagai *Validation Set*, dan 10% sebagai *Testing Set*, dimana pada setiap Set, terdapat 9 Label Class Label Class yaitu: *Black_Sea_Sprat*, *Gilt_Head_Bream*, *Horse_Mackerel*, *Red_Mullet*, *Red_Sea_Bream*, *Sea_Bass*, *Shrimp*, *Striped_Red_Mullet*, *Trout*..
 
## Main
1. Run Script Settings To Extracted Dataset (DatasetScript.ipynb)
2. Run IKAN_UAP.ipynb

## Preprocessing & Modelling

### CNN Model ✨
**Preprocessing**

Preprocessing yang dilakukan antara lain adalah *resizing* **(128,128)**, lalu *rescale / normalization* dengan rentang **1./255**, dilanjut dengan melakukan *splitting* dataset menjadi 3 *(Training, Validation, dan Testing)* sesuai dengan penjelasan pada Dataset.

**Modelling**

Hasil dari CNN Model yang telah dibangun adalah sebagai berikut :

![image](assets/3_ModelSummaryCNN.png)

**Model Evaluation**

Berikut adalah hasil dari fitting CNN Model yang telah dibangun :

![image](assets/4_CNNCurve.png)

a. Plot diatas menunjukkan bahwa training acc dapat stabil diatas **95%**, namun validation acc nya mengalami fluktuatif acc pada rentang **70 hingga 90%**, 
b. Plot diatas menunjukkan bahwa loss dari training set stabil di **1.0**, sedangkan val_loss nya mengalami fluktuatif dengan rentang loss antara **0.4 hingga 1.0**.

![image](assets/5_CNNCF.png)

Gambar diatas merupakan *Classification Report* dari Model setelah dilakukan *predict* terhadap *Testing Set*. Dapat dilihat bahwa Akurasinya mencapai **95%** dengan hasil prediksi pada label *'Rock'* dapat sempurna di **100% acc**.

### Inception-V3 Model ✨
**Preprocessing**

Preprocessing yang dilakukan antara lain adalah resizing **(299,299)** sesuai rekomendasi Inception-V3, lalu rescale / normalization dengan rentang 1./255, lalu melakukan augmentasi dengan parameter seperti *sheer_range* yang diatur ke **0.2**, *zoom_range* diatur ke **0.2**, dan *horizontal_flip*. Setelah augmentasi selesai dilakukan, langkah terakhir adalah *splitting* dataset menjadi 3 *(Training, Validation, dan Testing)* sesuai dengan penjelasan pada Dataset.

**Modelling & Evaluation**

Berikut hasil dari Model setelah dilakukan *Fine-Tuning* menggunakan dataset RPS :

![image](assets/6_MobileCurve.png)

a. Plot diatas menunjukkan bahwa *training_acc* stabil mendekati **100%**, namun *val_acc* nya cuma mencapai **94%**, hal ini menjadi indikasi bahwa model mengalami *overfitting*.
b. Dapat dilihat pada plot loss diatas. *Training dan Val Loss* sama - sama turun, namun val_loss cenderung lebih tinggi dibanding training_loss nya. Hal ini mungkin saja disebabkan karena terjadi *Overfitting* pada Model dan perlu dilakukan *tuning* lebih lanjut untuk menghilangkan *Overfit*.

![image](assets/7_MobileCF.png)

Gambar diatas menunjukkan *Classification Report* dari Model setelah dilakukan predict terhadap *Testing Set*. Terlihat bahwa Model sangat akurat dan lebih baik dari CNN Model dalam generalisasi data dengan Akurasi tepat **100%**.

## Local Web Deployment

### Tampilan HomePage

![image]()

### Tampilan HomePage Setelah Upload Image

![image]()

### Tampilan Prediction Result

![image]()

## Author 👨‍💻

- [@HaidarZ](https://github.com/hazarddrips)