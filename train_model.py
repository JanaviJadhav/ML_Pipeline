import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load Preprocessed Data
X = pd.read_csv("data/processed_data.csv")
y = pd.read_csv("data/target.csv")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f" Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Save Model
pickle.dump(model, open("model/model.pkl", "wb"))
print(" Model Training Completed & Saved.")
