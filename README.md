# 🔍 Industrial Casting Defect Detection API

A complete end-to-end Machine Learning project that detects defects in industrial casting products using Computer Vision and Deep Learning. The system uses a custom Convolutional Neural Network (CNN) built with PyTorch and exposes predictions through a FastAPI REST API.

---

## 🚀 Features

* Custom CNN model built from scratch using PyTorch
* Automated casting defect detection
* FastAPI-powered REST API
* Real-time image predictions
* Confidence score for every prediction
* Local inference support
* Production-ready API endpoint

---

## 🛠️ Tech Stack

### Machine Learning

* PyTorch
* Torchvision

### Backend

* FastAPI
* Uvicorn

### Image Processing

* Pillow (PIL)

### Visualization

* Matplotlib

---

## 📂 Project Structure

```text
Industrial-Defect-Detection-API/
│
├── dataset.py
├── model.py
├── train.py
├── predict.py
├── main.py
├── defect_model.pth
│
├── data/
│   ├── train/
│   └── test/
│
└── README.md
```

---

## 🧠 Model Architecture

The project uses a custom CNN called `DefectDetectorCNN`.

### Architecture

```text
Input Image (224x224x1)
        │
        ▼
Conv2D (1 → 16)
        │
      ReLU
        │
    MaxPool
        │
        ▼
Conv2D (16 → 32)
        │
      ReLU
        │
    MaxPool
        │
        ▼
Flatten
        │
        ▼
Linear Layer
(100352 → 128)
        │
      ReLU
        │
        ▼
Linear Layer
(128 → 1)
        │
     Sigmoid
        │
        ▼
Defective / OK
```

---

## 📊 Dataset Structure

Organize your dataset as follows:

```text
data/
├── train/
│   ├── def_front/
│   └── ok_front/
│
└── test/
    ├── def_front/
    └── ok_front/
```

Update the dataset paths in `dataset.py` according to your local machine.

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/industrial-defect-detection-api.git

cd industrial-defect-detection-api
```

### Install Dependencies

```bash
pip install torch torchvision fastapi uvicorn pillow matplotlib
```

Or

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

Run:

```bash
python train.py
```

Training Details:

* Loss Function: Binary Cross Entropy Loss
* Optimizer: Adam
* Epochs: 5
* Learning Rate: 0.001

After training completes:

```text
defect_model.pth
```

will be generated automatically.

---

## 🌐 Run the API Server

Start FastAPI:

```bash
uvicorn main:app --reload
```

Server URL:

```text
http://localhost:8000
```

Swagger Documentation:

```text
http://localhost:8000/docs
```

---

## 🔮 API Endpoint

### POST /predict

Upload an image and receive a prediction.

#### Request

* Method: POST
* Content-Type: multipart/form-data
* Input: Image File

#### Example Response

```json
{
  "filename": "cast_def_0_118.jpeg",
  "prediction": "DEFECTIVE",
  "confidence_score": 0.9847
}
```

---

## 🖥️ Local Prediction

Run:

```bash
python predict.py
```

Example Output:

```text
Result: DEFECTIVE
Confidence: 98.47%
```

---

## 🔄 Workflow

```text
Casting Image
      │
      ▼
Image Preprocessing
(Resize + Grayscale)
      │
      ▼
CNN Model
      │
      ▼
Prediction Score
      │
      ▼
DEFECTIVE / OK
      │
      ▼
JSON Response
```

---

## 📈 Future Improvements

* Docker Deployment
* AWS EC2 Deployment
* Render Deployment
* Model Quantization
* CI/CD Pipeline
* React Frontend
* Mobile Application
* Grad-CAM Visualization

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push changes

```bash
git push origin feature-name
```

5. Create a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Srihari**

GitHub: https://github.com/yourusername

LinkedIn: https://linkedin.com/in/yourprofile

---

⭐ If you found this project useful, please give it a star.
