
# Atrial Fibrillation Detection Using Deep Learning

This repository provides a comprehensive solution for detecting atrial fibrillation (AF) from ECG signals using advanced signal processing and deep learning techniques. The pipeline includes preprocessing raw ECG data, balancing the dataset, building a robust deep learning model, and evaluating its performance using various metrics.

---

## Features
- **Signal Preprocessing**:
  - High-pass and low-pass filtering for noise removal.
  - Normalization for consistent scaling.

- **Dataset Handling**:
  - Segmenting ECG signals into fixed-length windows.
  - Synthetic data generation using SMOTE to balance imbalanced classes.

- **Deep Learning Model**:
  - **Conv1D** layers for feature extraction.
  - **LSTM** layers for temporal pattern learning.
  - **Attention Mechanism** for feature weighting.
  - Regularization techniques (Dropout, Batch Normalization).

- **Evaluation Metrics**:
  - Accuracy, Precision-Recall AUC, F1-Score, MCC, Balanced Accuracy.
  - Confusion Matrix for classification insights.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
Install dependencies:

pip install wfdb numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow
Download the AF database from PhysioNet:


wget -r -N -c -np https://physionet.org/files/afdb/1.0.0/
Usage
Ensure the required AF database is downloaded into ./physionet.org/files/afdb/1.0.0/.

Run the main script:

python main.py
View performance metrics and visualizations:

Training/Validation accuracy and loss.
Confusion matrix.
Precision-recall and ROC curves.

File Structure

.
├── main.py                # Main script to execute the pipeline
├── README.md              # Project documentation
└── physionet.org/         # Directory for AF database
Future Improvements
Integration of more complex architectures like Transformers.
Handling additional annotations (e.g., AFL, J).
Automated hyperparameter tuning.
License
This repository is open-source under the MIT License.

Acknowledgments
PhysioNet for providing the AF database: https://physionet.org/
TensorFlow and Keras for deep learning tools.
markdown
Copy code
