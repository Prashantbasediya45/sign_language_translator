# train_model.py - FIXED VERSION
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
print("Loading data...")
df = pd.read_csv('sign_data.csv', encoding='utf-8')
print(f"Loaded {len(df)} samples")

# Prepare features and labels
X = df.drop('label', axis=1).values
y = df['label'].values

# Check which signs we have
signs = ['Hello', 'I Love You', 'No', 'Please', 'Thank You', 'Yes']
unique_labels = np.unique(y)
signs_present = [signs[i] for i in unique_labels]

print(f"\nSigns in dataset: {signs_present}")
print(f"Number of classes: {len(unique_labels)}")

# Show distribution
from collections import Counter
label_counts = Counter(y)
print("\nSamples per sign:")
for label in sorted(label_counts.keys()):
    print(f"  {signs[label]}: {label_counts[label]} samples")

# Check if we have enough data
if len(unique_labels) < 2:
    print("\n⚠️  WARNING: You need at least 2 different signs to train!")
    print("Please run collect_data.py and record more signs.")
    exit()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train model
print("\nTraining Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"Model Accuracy: {accuracy*100:.2f}%")
print(f"{'='*50}")

# Detailed report - only for signs present
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=signs_present, labels=unique_labels))

# Save model
joblib.dump(model, 'sign_model.pkl')
print("\n✓ Model saved to sign_model.pkl")

# Save sign labels mapping
import json
label_mapping = {int(label): signs[label] for label in unique_labels}
with open('sign_labels.json', 'w') as f:
    json.dump(label_mapping, f)
print("✓ Labels saved to sign_labels.json")

print("\n" + "="*50)
if len(unique_labels) < 6:
    print("⚠️  NOTE: You've only trained on {}/{} signs.".format(len(unique_labels), 6))
    print("Missing signs:", [signs[i] for i in range(6) if i not in unique_labels])
    print("Collect more data for better results!")
print("="*50)