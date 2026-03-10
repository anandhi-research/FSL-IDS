import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_dataset(path):

    data = pd.read_csv(path)

    if "label" not in data.columns:
        raise ValueError("Dataset must contain a 'label' column")

    X = data.drop("label", axis=1)
    y = data["label"]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.66, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
