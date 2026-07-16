import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. LOAD AND CLEAN DATASET
# ==========================================
# Load the CSV file
df = pd.read_csv("heart.csv")

# Remove duplicate rows if any exist
if df.duplicated().any():
    print(f"Removing {df.duplicated().sum()} duplicate rows...")
    df = df.drop_duplicates()

# ==========================================
# 2. FEATURE SPLITTING & TRAIN-TEST SPLIT
# ==========================================
# Separate features (X) and target variable (y)
X = df.drop(columns=["output"])
y = df["output"]

# Split into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==========================================
# 3. FEATURE SCALING
# ==========================================
# Scale features to have a mean of 0 and standard deviation of 1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. MODEL TRAINING
# ==========================================
# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ==========================================
# 5. GENERATE PREDICTIONS & TEXT EVALUATION
# ==========================================
# Run predictions on the test set
y_pred = model.predict(X_test_scaled)

# Print accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}\n")

# Print textual classification reports
print("--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred))

# Get feature importances
importances = model.feature_importances_
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=True)

print("\n--- Feature Importances (Sorted) ---")
print(feature_importances.sort_values(ascending=False))

# ==========================================
# 6. VISUALIZATION (SIDE-BY-SIDE PLOTS)
# ==========================================
# Create a figure with two subplots side-by-side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Confusion Matrix Heatmap
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    ax=axes[0],
    xticklabels=['Less Risk (0)', 'More Risk (1)'],
    yticklabels=['Less Risk (0)', 'More Risk (1)']
)
axes[0].set_title('Confusion Matrix Heatmap', fontsize=14, pad=10)
axes[0].set_xlabel('Predicted Label', fontsize=12)
axes[0].set_ylabel('True Label', fontsize=12)

# Plot 2: Horizontal Bar Chart of Feature Importances
feature_importances.plot(kind='barh', color='skyblue', ax=axes[1])
axes[1].set_title('Feature Importance Analysis', fontsize=14, pad=10)
axes[1].set_xlabel('Importance Score', fontsize=12)
axes[1].set_ylabel('Clinical Features', fontsize=12)

# Adjust layout and render
plt.tight_layout()
plt.show()
