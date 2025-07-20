# save_models.py
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# Create models directory
os.makedirs("models", exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv("Colorectal Cancer Patient Data.csv")

# Drop unnecessary columns
df.drop(columns=["Unnamed: 0", "ID_REF"], inplace=True)

# Drop rows with missing target values
df = df.dropna(subset=["DFS event"])

# Fill missing values for other columns
imputer = SimpleImputer(strategy="most_frequent")
df[df.columns] = imputer.fit_transform(df)

# Extract features and target
X = df.drop(columns=["DFS event"])
y = df["DFS event"].astype(float).astype(int)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n==== Encoding Categorical Variables ====")
# Combine features and target back temporarily for encoding
train_data = X_train.copy()
train_data['DFS event'] = y_train

test_data = X_test.copy()
test_data['DFS event'] = y_test

# Encode categorical columns using only training data to fit
label_encoders = {}
categorical_cols = ["Dukes Stage", "Gender", "Location"]

for col in categorical_cols:
    if col in train_data.columns:
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col])
        test_data[col] = le.transform(test_data[col])
        label_encoders[col] = le

        # Show mapping for better understanding
        mapping = {original: encoded for original, encoded in zip(le.classes_, range(len(le.classes_)))}
        print(f"Encoded {col}: {mapping}")

# Now separate features and labels again
X_train = train_data.drop('DFS event', axis=1)
y_train = train_data['DFS event']
X_test = test_data.drop('DFS event', axis=1)
y_test = test_data['DFS event']

print("\n==== Scaling Numerical Features ====")
# Scale numerical features using only training data
numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
print(f"Scaled {len(numeric_cols)} numerical features")

print("\n==== Feature Selection ====")
# Feature selection using only training data
selector = SelectKBest(f_classif, k=6)
selector.fit(X_train, y_train)
selected_indices = selector.get_support(indices=True)
selected_features = X_train.columns[selected_indices]

print(f"Selected {len(selected_features)} features:")
for i, feature in enumerate(selected_features):
    print(f"{i + 1}. {feature}")

# Apply feature selection to both training and test data
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

print("\n==== Model Training and Evaluation ====")
# Train KNN
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train_selected, y_train)

# KNN Model Evaluation
y_pred_knn = knn.predict(X_test_selected)
print("\n==== KNN Model Evaluation ====")
print(f"Training accuracy: {knn.score(X_train_selected, y_train):.4f}")
print(f"Testing accuracy: {knn.score(X_test_selected, y_test):.4f}")

# Print confusion matrix for KNN
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("\nConfusion Matrix (KNN):")
print("                 Predicted")
print("                 No Cancer | Cancer")
print(f"Actual No Cancer |    {cm_knn[0][0]}     |   {cm_knn[0][1]}")
print(f"Actual Cancer    |    {cm_knn[1][0]}     |   {cm_knn[1][1]}")

# Print classification report for KNN
print("\nClassification Report (KNN):")
print(classification_report(y_test, y_pred_knn, target_names=['No Cancer', 'Cancer']))

# Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train_selected, y_train)

# Naive Bayes Model Evaluation
y_pred_nb = nb.predict(X_test_selected)
print("\n==== Naive Bayes Model Evaluation ====")
print(f"Training accuracy: {nb.score(X_train_selected, y_train):.4f}")
print(f"Testing accuracy: {nb.score(X_test_selected, y_test):.4f}")

# Print confusion matrix for Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb)
print("\nConfusion Matrix (Naive Bayes):")
print("                  Predicted")
print("                  No Cancer | Cancer")
print(f"Actual No Cancer |    {cm_nb[0][0]}     |   {cm_nb[0][1]}")
print(f"Actual Cancer    |    {cm_nb[1][0]}     |   {cm_nb[1][1]}")

# Print classification report for Naive Bayes
print("\nClassification Report (Naive Bayes):")
print(classification_report(y_test, y_pred_nb, target_names=['No Cancer', 'Cancer']))

# Save all models and objects
print("\nSaving models and objects to 'models/' directory...")
with open('models/knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)
with open('models/nb_model.pkl', 'wb') as f:
    pickle.dump(nb, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('models/selected_features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

print("\nModel files created successfully!")
print("Verify model files exist:")
for file in os.listdir('models'):
    print(f"- {file}: {os.path.getsize(f'models/{file}')} bytes")