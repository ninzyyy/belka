import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Load the balanced train data
print(f"\nLoading training data...")
df = pd.read_parquet("data/balanced_train.parquet")

# Extract the target values
print(f"\Pulling target data from training set...")
y = df["binds"].values

# Load the train data
print(f"\nLoading ecfp_train data...")
ecfp_train = np.load("../data/ecfp_train.npy")

print(f"\nLoading protein_train data...")
protein_train = np.load("../data/protein_embed_train.npy")

# Concatenate the molecule and protein features for the train set
print(f"\nConcatenating molecule and protein train data...")
X = np.concatenate((ecfp_train, protein_train), axis=1)

# Train the model
print(f"\nTraining the XGBClassifier model...")
model = XGBClassifier()
model.fit(X, y)

# Load the testing data
print(f"\nLoading testing data...")
test_df = pd.read_parquet("data/test.parquet")

print(f"\nLoading ecfp_test data...")
ecfp_test = np.load("data/ecfp_test.npy")

print(f"\nLoading protein_test data...")
protein_test = np.load("data/protein_embed_test.npy")

# Concatenate the molecule and protein features for the test set
print(f"\nConcatenating molecule and protein test data...")
X_test = np.concatenate((ecfp_test, protein_test), axis=1)

print(f"\nPredicting probabilities...")
probabilities = model.predict_proba(X_test)[:, 1]

print(f"\nGrabbing ids of row...")
ids = test_df["id"].values

print(f"\nCreating submission file dataframe...")
submission_df = pd.DataFrame({"id": ids, "binds": probabilities})

print(f"\nSaving submission file...")
submission_df.to_csv("data/2024_05_02.csv", index=False)

print(f"\n✅ Submission file saved!")
