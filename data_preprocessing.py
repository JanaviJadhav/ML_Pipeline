import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load Dataset
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", 
                 header=None, na_values=" ?", skipinitialspace=True)

# Define Column Names
df.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
              "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
              "hours-per-week", "native-country", "income"]

# Drop missing values
df.dropna(inplace=True)

# Convert Target Variable (<=50K: 0, >50K: 1)
df["income"] = df["income"].apply(lambda x: 1 if x == ">50K" else 0)

# Encode Categorical Variables
categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship",
                    "race", "sex", "native-country"]
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Define Features & Target
X = df.drop(columns=["income"])
y = df["income"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save Scaler
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

# Save Preprocessed Data
pd.DataFrame(X_scaled).to_csv("data/processed_data.csv", index=False)
pd.DataFrame(y).to_csv("data/target.csv", index=False)

print("Data Preprocessing Completed.")
