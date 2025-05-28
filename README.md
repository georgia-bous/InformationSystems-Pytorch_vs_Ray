# ⚙️ InformationSystems - PyTorch vs Ray

This project explores the capabilities of **PyTorch** and **Ray** in the context of machine learning and distributed systems.

We implemented three core tasks in Python:
- Binary Classification on CSV data
- Personalized PageRank on a graph
- Image Classification using the EMNIST dataset

The goal is to compare **performance**, **efficiency**, and **scalability** between the two frameworks.

---

## 📁 Repository Structure

├── code/ # Python implementations for each task

│ ├── PersonalisedPRPytorch.py # PageRank with PyTorch

│ ├── PersonalisedPRRay.py # PageRank with Ray

│ ├── classificationPytorch.py # CSV classification with PyTorch

│ ├── classificationRay.py # CSV classification with Ray

│ ├── emnistPytorch.py # EMNIST classification with PyTorch

│ ├── emnistRay.py # EMNIST classification with Ray

│ └── preprocessing.py # Preprocessing tools for input data

│
├── data/ # Contains a URL reference to datasets (not uploaded)

│
├── documents/ # Documentation and report

│ ├── InformationSystemsReport.pdf # Full report with results and conclusions

│ └── εκφώνηση.pdf # Project prompt/description
│
└── directions.txt # Instructions for running the system


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
