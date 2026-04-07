import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def train_model(data_path):

    print("Loading dataset...")

    df = pd.read_csv(data_path)

    print("Dataset shape:", df.shape)

    # convert target variable
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # encode categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    le = LabelEncoder()

    for col in categorical_cols:
        if col != "Attrition":
            df[col] = le.fit_transform(df[col])

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    print("Training model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    # create models folder if not exists
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/attrition_prediction_model.pkl")

    print("Model saved in models/attrition_prediction_model.pkl")


if __name__ == "__main__":

    train_model("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")