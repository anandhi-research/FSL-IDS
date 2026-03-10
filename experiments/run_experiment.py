from data.dataset_loader import load_dataset
from models.sae_model import build_sae
from models.hefce_model import HEFCE
from sklearn.metrics import accuracy_score, f1_score, recall_score

DATASET_PATH = "data/iotid20.csv"

X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(DATASET_PATH)

input_dim = X_train.shape[1]

sae = build_sae(input_dim)

sae.fit(
    X_train,
    X_train,
    epochs=10,
    batch_size=64
)

features_train = sae.predict(X_train)
features_test = sae.predict(X_test)

classifier = HEFCE()

classifier.fit(features_train, y_train)

pred = classifier.predict(features_test)

accuracy = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average="weighted")
recall = recall_score(y_test, pred, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Recall:", recall)
