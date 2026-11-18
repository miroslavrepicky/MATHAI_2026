"""
Tréning NN na vlastnom datasete (dataset.npz z extractor.py).

- delenie každej triedy 80 : 20 na tréning / test (seed 42), test sa netrénuje
- SGD po jednej vzorke, eta = 0.1, 300 epoch, pred každou epochou shuffle
- po tréningu: dopredné šírenie na teste + presnosť
"""
import numpy as np

from base_model import BaseModel
from layers import InputLayer, HiddenLayer, OutputLayer
from activations import Sigmoid, Softmax
from loss_funtions import CCE

CHARS = ["9", "H", "L", "P", "O", "T", "M", "5", "2", "Z"]   # poradie z tabuľky
SEED = 42
TEST_FRAC = 0.2
EPOCHS = 300
LR = 0.1
PRINT_EVERY = 10


def stratified_split(X, Y, test_frac=TEST_FRAC, seed=SEED):
    """Každú triedu zvlášť náhodne rozdelí ~80:20."""
    rng = np.random.RandomState(seed)
    labels = Y.argmax(axis=1)
    train_idx, test_idx = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        n_test = max(1, int(round(len(idx) * test_frac)))
        test_idx += list(idx[:n_test])
        train_idx += list(idx[n_test:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]


def predict(model, X):
    return np.array([np.argmax(model.forward(x)) for x in X])


def accuracy(model, X, Y):
    return float(np.mean(predict(model, X) == Y.argmax(axis=1)))


def train(model, X, Y, epochs=EPOCHS, lr=LR, loss=None, seed=SEED):
    model.loss = loss
    rng = np.random.RandomState(seed)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(X))              # shuffle pred každou epochou
        total = 0.0
        for i in order:
            out = model.forward(X[i])
            total += float(np.mean(loss.forward(out, Y[i])))
            model.backward(X[i], Y[i], lr)           # SGD: update po jednej vzorke
        if epoch == 1 or epoch % PRINT_EVERY == 0:
            print(f"Epoch {epoch:3d}/{epochs}  loss: {total / len(X):.5f}")


def main():
    data = np.load("dataset.npz")
    X, Y = data["X"], data["Y"]
    X_train, Y_train, X_test, Y_test = stratified_split(X, Y)
    print(f"Celkom {len(X)} vzoriek -> tréning {len(X_train)}, test {len(X_test)}")

    model = BaseModel()
    model.add_layer(InputLayer(784))
    model.add_layer(HiddenLayer(128, Sigmoid()))
    model.add_layer(OutputLayer(10, Softmax()))
    model.build()

    train(model, X_train, Y_train, loss=CCE())   # Softmax + cross-entropy

    print(f"\nPresnosť na tréningu: {accuracy(model, X_train, Y_train) * 100:.1f} %")
    pred = predict(model, X_test)
    true = Y_test.argmax(axis=1)
    acc = float(np.mean(pred == true))
    print(f"Presnosť na teste:    {acc * 100:.1f} %  ({int((pred == true).sum())}/{len(true)})")
    for p, t in zip(pred, true):
        if p != t:
            print(f"  zle: správne {CHARS[t]}, predikcia {CHARS[p]}")
    print("Požiadavka > 50 % splnená" if acc > 0.5 else "Požiadavka > 50 % NESPLNENÁ")


if __name__ == "__main__":
    main()
