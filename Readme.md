# QuantumYOLOv8n: Hybrid Quantum-Classical Object Detection

## 📌 Project Description

This project integrates a **Quantum Neural Network (QNN)** with **YOLOv8n (nano variant)** to create a **hybrid quantum-classical object detection model**.
The classical YOLOv8n backbone is enhanced by inserting a QNN block into its feature extraction pipeline, allowing the model to leverage quantum computing advantages such as richer feature representation and potential noise robustness.

The project demonstrates how **quantum computing** can be applied in **real-world computer vision** tasks like object detection, while still running efficiently on classical hardware.

## 🎥 Project Demo Video:

<video src="https://github.com/AgrimGusain/SignSense/raw/main/VID1.mp4" controls="controls" width="600">
  Your browser does not support the video tag.
</video>



---

## ⚡ Features

* YOLOv8n (Ultralytics) used as the baseline object detector
* Quantum Neural Network integrated as a feature extractor
* Built with **PennyLane + PyTorch**
* Compatible with **GPU + Quantum Simulator**
* Flexible: can run in full-classical mode (disable QNN) or hybrid mode

---

## 🛠️ Project Structure

```
QuantumYOLOv8n/
│── README.md               # Documentation
│── requirements.txt        # Dependencies
│── quantum_layer.py        # Custom QNN layer
│── models/
│   ├── yolo.py             # Modified YOLOv8n model with QNN integration
│   ├── common.py           # Utility layers
│   └── __init__.py
│── train.py                # Training script
│── val.py                  # Validation script
│── data/                   # Dataset folder
│── runs/                   # Training outputs
```

---

## 📦 Installation

1. Clone the repository

```bash
git clone https://github.com/your-username/QuantumYOLOv8n.git
cd QuantumYOLOv8n
```

2. Create a virtual environment

```bash
conda create -n quantumyolo python=3.10 -y
conda activate quantumyolo
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

### ✅ Recommended Library Versions

* `torch==2.1.0`
* `torchvision==0.16.0`
* `ultralytics==8.0.196`
* `pennylane==0.33.1`
* `pennylane-lightning==0.33.1`
* `numpy==1.26.0`

---

## ⚙️ How It Works

1. **Classical Backbone (YOLOv8n)**

   * Standard convolutional layers extract features from input images.

2. **Quantum Layer (QNN)**

   * Inserted inside the **`YOLOv8n Backbone (C2f block)`**.
   * Input features → Encoded into quantum states → Processed by QNN → Output features.
   * Implemented in `quantum_layer.py` using PennyLane.

3. **Detection Head**

   * Same as YOLOv8n. Predicts bounding boxes, object classes, and confidence scores.

---

## 🧑‍💻 Training

```bash
python train.py model=models/yolo.py data=data.yaml epochs=50 imgsz=640
```

---

## 🔍 Inference

```bash
python val.py model=runs/train/exp/weights/best.pt source=test_images/
```

---

## 📖 Example Quantum Layer (`quantum_layer.py`)

```python
import torch
import torch.nn as nn
import pennylane as qml

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.circuit = circuit
        self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)[:, :self.n_qubits]
        return self.qlayer(x)
```

---

## 🚀 Future Work

* Run on **real quantum hardware (IBM Q, Xanadu X-Series)**
* Compare YOLOv8n vs QuantumYOLOv8n performance
* Optimize quantum circuits for speed & noise tolerance

---

## 📜 Citation

If you use this project, please cite:

```bibtex
@misc{QuantumYOLOv8n2025,
  author       = Agrim Gusain,
  title        = SignSense,
  year         = 2025,
  url          = https://github.com/AgrimGusain/SignSense
}
```

---

Do you want me to also **add step-by-step instructions in the README about exactly where in YOLOv8n you modified the architecture to include the QNN (like which file and class to edit)?** That way, anyone reading the README can reproduce your setup directly.
