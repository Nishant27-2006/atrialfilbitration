# Install and download required datasets
!pip install wfdb
!wget -r -N -c -np https://physionet.org/files/afdb/1.0.0/

import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report, confusion_matrix, f1_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, LSTM, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import scipy.signal as sp

# Signal Preprocessing
def preprocess_signal(signal, fs=250):
    b, a = sp.butter(1, 0.5 / (fs / 2), btype='highpass')
    filtered_signal = sp.filtfilt(b, a, signal)
    b, a = sp.butter(1, 50 / (fs / 2), btype='lowpass')
    filtered_signal = sp.filtfilt(b, a, filtered_signal)
    normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
    return normalized_signal

# Dataset Preprocessing
def preprocess_dataset(files, record_base_path, segment_length=5000):
    X, y = [], []
    class_distribution = {"Normal": 0, "Disease": 0}
    for file in files:
        file_base_path = os.path.join(record_base_path, file)
        if not os.path.exists(f"{file_base_path}.dat") or not os.path.exists(f"{file_base_path}.hea"):
            continue
        try:
            record = wfdb.rdrecord(file_base_path)
            signal = preprocess_signal(record.p_signal[:, 0])
            label = 0 if 'normal' in file.lower() else 1
            class_name = 'Normal' if label == 0 else 'Disease'
            class_distribution[class_name] += 1
            segments = [signal[i:i + segment_length] for i in range(0, len(signal) - segment_length, segment_length)]
            if segments:
                X.append(np.array(segments))
                y.extend([label] * len(segments))
        except Exception as e:
            print(f"Error processing {file_base_path}: {e}")
            continue
    print(f"Class Distribution Before Balancing: {class_distribution}")
    X = np.vstack(X)
    y = np.array(y)
    if len(set(y)) < 2:
        print("[INFO] Generating synthetic data for the missing class...")
        target_class = 1 if 0 not in y else 0
        synthetic_X, synthetic_y = generate_synthetic_data(X, target_class, n_samples=10000)
        X = np.vstack((X, synthetic_X))
        y = np.concatenate((y, synthetic_y))
    return X, y

# Generate Synthetic Data
def generate_synthetic_data(X, target_class, n_samples):
    synthetic_X = []
    for _ in range(n_samples):
        sample = X[np.random.randint(len(X))].copy()
        sample += np.random.normal(0, 0.1, sample.shape)
        synthetic_X.append(sample)
    synthetic_y = [target_class] * n_samples
    return np.array(synthetic_X), np.array(synthetic_y)

# Build Advanced Model
def build_advanced_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.4)(x)
    attention = Attention()([x, x])
    x = Flatten()(attention)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Plot Class Distribution
def plot_class_distribution(y, title):
    class_counts = np.bincount(y)
    plt.bar(['Normal', 'Disease'], class_counts, color=['blue', 'red'])
    plt.title(title)
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.show()

# Plot Training History
def plot_training_history(history):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Normal', 'Disease'], yticklabels=['Normal', 'Disease'])
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Additional Visualizations
def plot_roc_curve(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR Curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.show()

def calculate_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    print("\nEvaluation Metrics:")
    print(f"F1 Score: {f1:.2f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")
    print(f"Balanced Accuracy: {balanced_acc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Disease']))

# Main Function with Additional Visualizations
def main():
    record_base_path = './physionet.org/files/afdb/1.0.0/'
    files = [file.split('.')[0] for file in os.listdir(record_base_path) if file.endswith('.hea')]
    X, y = preprocess_dataset(files, record_base_path)
    plot_class_distribution(y, "Class Distribution Before Balancing")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    plot_class_distribution(y_train, "Training Set Class Distribution")
    
    model = build_advanced_model((X_train.shape[1], 1))
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30, batch_size=64,
        callbacks=[lr_scheduler, early_stopping]
    )
    
    plot_training_history(history)
    
    y_pred_prob = model.predict(X_val).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    plot_confusion_matrix(y_val, y_pred)
    plot_roc_curve(y_val, y_pred_prob)
    plot_precision_recall_curve(y_val, y_pred_prob)
    calculate_metrics(y_val, y_pred)

if __name__ == "__main__":
    main()
