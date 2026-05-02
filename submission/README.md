# Brain Tumor Classification - Submission

## Deskripsi
Proyek klasifikasi gambar untuk mendeteksi jenis tumor otak menggunakan CNN.

## Dataset
**BRISC 2025 - Brain Tumor Classification**
- Sumber: Kaggle (briscdataset/brisc2025)
- Kelas: glioma, meningioma, pituitary, no_tumor
- Split: 70% train / 15% validation / 15% test (stratified per kelas)

## Arsitektur Model
Sequential CNN dengan 4 blok Conv2D:
- Block 1: Conv2D(32) + BatchNorm + MaxPooling2D
- Block 2: Conv2D(64) + BatchNorm + MaxPooling2D
- Block 3: Conv2D(128) + BatchNorm + MaxPooling2D
- Block 4: Conv2D(256) + BatchNorm + MaxPooling2D
- Head: GlobalAveragePooling2D → Dense(512) → Dropout(0.4) → Dense(256) → Dropout(0.3) → Softmax(4)
- Input size: 224x224x3

## Struktur Direktori
```
submission/
├── tfjs_model/
│   ├── group1-shard1of1.bin
│   └── model.json
├── tflite/
│   ├── model.tflite
│   └── label.txt
├── saved_model/
│   ├── saved_model.pb
│   └── variables/
├── notebook.ipynb
├── README.md
└── requirements.txt
```

## Cara Menjalankan
1. Install dependencies: `pip install -r requirements.txt`
2. Jalankan notebook: `notebook.ipynb`
