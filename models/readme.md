# 📦 Model Weights Setup

Due to size limitations, the trained model weights are **not included directly in this repository**.

## 🔽 Download Model Weights

Please download the model files from Google Drive:

👉 https://drive.google.com/drive/folders/1XOnn_elc1FOV4RE-k7SYh_c_IBxBYNqH?usp=sharing

---

## 📁 Installation Steps

1. Download all files from the Google Drive folder.
2. Extract the downloaded `.zip` file.
3. Place the extracted contents into the following directory:

```
models/
```

After extraction, your project structure should look like this:

```
project-root/
│
├── data/
├── models/
│   ├── best_bilstm1d.pt
│   ├── best_cnn1d.pt
│   ├── best_lstm1d.pt
│   ├── best_resnet1d.pt
│   ├── best_rnn1d.pt
│   ├── lr_best.pkl
│   ├── rf_best.pkl
│   ├── scaler.pkl
│   ├── test_predictions.pkl
│   └── val_predictions.pkl
│
├── notebooks/
├── reports
├── src/
└── README.md
```

---

## ⚠️ Notes

- Ensure all model files are placed **directly inside the `models/` folder**.
- Do **not rename** the files, as the code expects specific filenames.

---
