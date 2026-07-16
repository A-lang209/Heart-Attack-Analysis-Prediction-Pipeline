# Heart-Attack-Analysis-Prediction-Pipeline

An end-to-end machine learning pipeline built in Python to analyze cardiovascular health markers and predict heart attack risk. This project uses a Random Forest Classifier to classify patients as low-risk (`0`) or high-risk (`1`) based on standard clinical indicators.

## 📌 Project Overview
The objective of this project is to build a supervised binary classification model that ingests clinical measurements (like resting blood pressure, cholesterol levels, maximum heart rate, and chest pain indicators) to evaluate a patient's risk profile. 

In healthcare, missing a high-risk patient (a False Negative) is highly critical. Therefore, this pipeline evaluates the model using detailed confusion matrices and focus on recall alongside overall accuracy.

## 📊 Dataset Features
The dataset consists of 303 patient records with 13 input features and 1 binary target:
- **`age`**: Patient's age in years.
- **`sex`**: Gender ($1 = \text{Male}$, $0 = \text{Female}$).
- **`cp`**: Chest pain type (Typical Angina, Atypical Angina, Non-Anginal, Asymptomatic).
- **`trtbps`**: Resting blood pressure (in mm Hg).
- **`chol`**: Serum cholesterol level (in mg/dL).
- **`fbs`**: Fasting blood sugar ($1 = \text{True}$ if $> 120 \text{ mg/dL}$).
- **`restecg`**: Resting electrocardiographic results.
- **`thalachh`**: Maximum heart rate achieved during exercise.
- **`exng`**: Exercise-induced angina ($1 = \text{Yes}$, $0 = \text{No}$).
- **`oldpeak`**: ST depression induced by exercise relative to rest.
- **`slp`**: Slope of the peak exercise ST segment.
- **`caa`**: Number of major vessels (0-3) colored by fluoroscopy.
- **`thall`**: Thalassemia blood disorder check.
- **`output` (Target)**: Risk level ($0 = \text{Less Risk}$, $1 = \text{More Risk}$).

## ⚙️ Key Methodology

### 1. Data Cleaning
- Automatic identification and removal of duplicate rows to prevent data leakage.

### 2. Feature Scaling
To prevent features with large raw values (like cholesterol or resting blood pressure) from dominating features on smaller scales (like ST depression), we apply Z-score standardization:
$$z = \frac{x - \mu}{\sigma}$$
- **Crucial Practice:** The `StandardScaler` is fitted *only* on the training data (`fit_transform`) and then applied to the test data (`transform`). This guarantees that no statistical data from our test set leaks into our training phase.

### 3. Model Architecture
We train a **Random Forest Classifier** (`n_estimators=100`). This ensemble learning technique builds multiple decision trees and aggregates their predictions, which significantly improves generalization and prevents overfitting.


📈 Performance & Results
The model achieves strong classification performance:

Overall Accuracy: ~80%

Recall for High-Risk Patients: ~91% (catching 30 out of 33 actual high-risk patients in the test set, ensuring a low False Negative rate).

Visualizations Included:

Confusion Matrix Heatmap: Displays exact counts of True Positives, True Negatives, False Positives, and False Negatives.

Feature Importance Plot: Shows that Chest Pain Type (cp), Maximum Heart Rate (thalachh), and ST depression (oldpeak) are the strongest predictors of heart attack risk.
