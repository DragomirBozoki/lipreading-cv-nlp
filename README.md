Download first 22 speakers and alignments: 

https://drive.google.com/file/d/1fDsKDHVsStDtCrt2fU08uSac-Y1d_ImT/view?usp=drive_link

# 🧠 Lipreading Neural Network – Visual Speech Recognition from Video

This project is the result of my **Bachelor thesis** and presents a complete deep learning pipeline for **visual speech recognition (lipreading)** using video recordings **without any audio input**.

It implements a full system for automatic transcription of spoken sentences by analyzing only **lip movements**. The model is trained and evaluated on the [GRID corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/), a standardized dataset used in visual speech research.

---

## 🎯 Objective

To build a deep neural network that can recognize spoken language based solely on video data, enabling robust and silent speech interfaces for accessibility, human-computer interaction, and noisy environments.

---

## 🗃 Dataset: GRID Corpus

- 34 speakers (18 male, 16 female), each uttering 1,000 sentences.
- Sentences follow a fixed grammar:
hile avoiding overfitting.

- First 22 speakers used for training/testing in this project.
- Download link for dataset (preprocessed subset):  
[📦 Download (Google Drive)](https://drive.google.com/file/d/1fDsKDHVsStDtCrt2fU08uSac-Y1d_ImT/view?usp=drive_link)

---

## 🧪 Preprocessing Pipeline

All preprocessing is custom-built using `dlib`, `OpenCV`, and `TensorFlow`:

- **Face detection** per frame using dlib.
- **Lip region extraction** based on facial landmarks (points 48–68).
- **Grayscale conversion** and resizing to 64×64.
- **Standardization** across frames (mean/std).
- **Label extraction** from `.align` files (excluding silences).
- **Tokenization** using a 41-character vocabulary (`a-z`, `'`, `?`, `!`, `1–9`, space).

Input tensors: `(75, 64, 64, 1)`  
Label tensors: `(40,)`

---

## 🧠 Model Architecture

Custom 3D-CNN + BiGRU model implemented in TensorFlow/Keras:

| Layer Type     | Configuration                         |
|----------------|----------------------------------------|
| Conv3D (x3)    | Filters: 128 → 256 → 64, Kernel: 3x3x3 |
| MaxPool3D      | Pooling: (1,2,2) after each conv block |
| TimeDistributed| Flatten over spatial dims              |
| BiGRU (x2)     | 128 units each, dropout 0.5            |
| Dense          | 41-class softmax output                |

- Output shape: `(batch_size, 75, 41)`
- Loss: **CTC Loss** (`ctc_batch_cost`)
- Optimizer: **Adam**, LR = 0.0001
- LR Scheduler: flat for 50 epochs → exponential decay every 50

---

## ⚙️ Training Strategy

- **500 epochs** total.
- Speaker changes every 50 epochs to improve generalization.
- Custom callbacks:
- `ProduceExample`: shows predictions after each epoch.
- `SaveHistoryCallback`: logs metrics every 5 epochs.
- `ModelCheckpoint`: saves weights to `/Weights`.

---

## ✅ Results & Use Cases

The model performs well on unseen speakers and shows potential for:

- Real-time webcam lipreading
- Assistive technologies for the hearing-impaired
- Speech interface in silent or noisy environments
- Integration into multimodal voice assistants (e.g. fallback when audio is missing)

---

## 📂 Project Structure

lipreading-cv-nlp/
├── model.py # CNN + BiGRU model definition
├── preprocessing.py # Preprocessing, alignment, vocab
├── train.py # Training loop & callbacks
├── datapipeline.py # tf.data batching and setup
├── Weights/ # Checkpointed weights
├── history_logs/ # Loss/accuracy logs
└── README.md


---

## 🚀 Future Work

- Add real-time inference from webcam (OpenCV + live model).
- Improve robustness using transformer-based language models.
- Combine with intent recognition for audio-free voice assistants.
- Deploy on edge devices (Raspberry Pi, Jetson Nano).

---

## 🧑‍💻 Author

**Dragomir Bozoki**  
Bachelor in Biomedical Engineering – Final Year Project  

---

## 📜 License

This project is licensed under the MIT License.
