import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Binary cross-entropy loss
def compute_loss(y, y_hat):
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# Training function
def train_nn(X, y, hidden_dim=16, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape

    # Set seed for reproducible weights
    np.random.seed(41)

    # He initialization for weights
    W1 = np.random.randn(n_features, hidden_dim) * np.sqrt(2. / n_features)
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2. / hidden_dim)
    b2 = np.zeros((1, 1))

    losses = []

    for epoch in range(epochs):
        # Forward pass
        Z1 = X @ W1 + b1      # (n, hidden)
        A1 = relu(Z1)
        Z2 = A1 @ W2 + b2     # (n, 1)
        y_hat = sigmoid(Z2)

        # Loss
        loss = compute_loss(y, y_hat)
        losses.append(loss)

        # Backpropagation
        dZ2 = y_hat - y.reshape(-1, 1)           # (n, 1)
        dW2 = A1.T @ dZ2 / n_samples
        db2 = np.mean(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * relu_derivative(Z1)          # (n, hidden)
        dW1 = X.T @ dZ1 / n_samples
        db1 = np.mean(dZ1, axis=0, keepdims=True)

        # Update weights
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch}, Loss = {loss:.4f}")

    return (W1, b1, W2, b2), losses

# Prediction function
def predict_nn(X, params):
    W1, b1, W2, b2 = params
    A1 = relu(X @ W1 + b1)
    y_hat = sigmoid(A1 @ W2 + b2)
    return (y_hat >= 0.5).astype(int)

def main():
    # Load the dataset with semicolon separator
    df = pd.read_csv("student-mat.csv", sep=";")

    # Create binary target column (1 = pass, 0 = fail) using final grade G3
    df['pass'] = (df['G3'] >= 10).astype(int)

    # Just drop G3, keep G1 and G2
    df = df.drop(['G3'], axis=1)

    # Split into features (X) and labels (y)
    X = df.drop('pass', axis=1)
    y = df['pass']

    # One-hot encode categorical variables (drop_first=True avoids dummy variable trap)
    X = pd.get_dummies(X, drop_first=True)

    # Convert to NumPy arrays
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    X_np = X_np.astype(np.float64)  # Ensure x is numeric
    y_np = y_np.astype(np.float64)  # Ensure y is numeric

    # Normalize the features
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_np)  # normalize entire dataset

    epochs = 1000
    lr = 0.05
    hidden_dim = 32

    # Train/Validation/Test split (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(X_np, y_np, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train the neural network
    params, losses = train_nn(X_train, y_train, hidden_dim, lr, epochs)

    # Predict and evaluate
    y_val_pred = predict_nn(X_val, params)

    print("Neural Network")
    print(f"Epochs: {epochs}, Learning Rate: {lr}, Hidden Dim: {hidden_dim}")

    # Print validation accuracy
    val_accuracy = np.mean(y_val_pred.flatten() == y_val)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Print predicted and actual counts
    num_predicted_pass = np.sum(y_val_pred == 1)
    num_predicted_fail = np.sum(y_val_pred == 0)
    total = len(y_val_pred)
    print(f"Predicted to pass: {num_predicted_pass} / {total}")
    print(f"Predicted to fail: {num_predicted_fail} / {total}")

    actual_pass = np.sum(y_val == 1)
    actual_fail = np.sum(y_val == 0)

    print(f"Actual passes: {actual_pass} / {total}")
    print(f"Actual fails: {actual_fail} / {total}")

    # Draw a plot of the learning curve of the training cross-entropy loss
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title('Training Loss over Epochs (Neural Network)')
    plt.plot(range(epochs), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('nn_training_loss.jpg')

    # Draw a confusion matrix for the validation set
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_val, y_val_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Fail", "Pass"])
    disp.plot()
    plt.title("Neural Network Confusion Matrix")
    plt.savefig("nn_confusion_matrix.jpg")


main()