# ⚙️ InformationSystems - PyTorch vs Ray

This project explores the capabilities of **PyTorch** and **Ray** in the context of machine learning and distributed systems.

We implemented three core tasks in Python:
- Binary Classification on CSV data
- Personalized PageRank on a graph
- Image Classification using the EMNIST dataset

The goal is to compare **performance**, **efficiency**, and **scalability** between the two frameworks.



---


---

## 🧪 Tasks & Implementations

### 🔹 Binary Classification (CSV Data)
- **Files:** `classificationPytorch.py`, `classificationRay.py`
- Compares model training on structured CSV data using both frameworks.

### 🔹 Personalized PageRank
- **Files:** `PersonalisedPRPytorch.py`, `PersonalisedPRRay.py`
- Implements the PageRank algorithm on graph data, measuring distributed performance.

### 🔹 Image Classification (EMNIST Dataset)
- **Files:** `emnistPytorch.py`, `emnistRay.py`
- Classifies handwritten characters and digits using the EMNIST dataset.

### 🔹 Data Preprocessing
- **File:** `preprocessing.py`
- Shared utility for preparing data formats suitable for training.

---

## 📊 Report & Results

For detailed analysis of the experimental setup, evaluation metrics, and performance comparisons:

📄 **[Read the full report →](documents/InformationSystemsReport.pdf)**

---

## 📦 Requirements

Install necessary Python libraries using:

```bash
pip install torch ray numpy pandas scikit-learn matplotlib

---

## 🚀 How to Run

Follow these steps to run the project:

1. **Download Data**  
   Go to the `data/` folder and follow the provided URL to download the datasets (this might take a while).

2. **Set Up Environment**  
   Install **Ray**, **PyTorch**, and other dependencies. Follow environment setup instructions as outlined in the first section of `InformationSystemsReport.pdf`.

3. **Preprocess Data (for Binary Classification)**  
   Before running the binary classification task, preprocess the raw data by running:

   ```bash
   python3 preprocessing.py
