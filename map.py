import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary cross-entropy loss
def compute_loss(y, y_pred):
    # Add small epsilon to avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# Gradient descent update step
def update_weights(X, y, weights, lr, lambda_):
    m = X.shape[0]
    y_pred = sigmoid(X @ weights)

    # Add regularization term to gradient (but not for bias)
    reg_term = (lambda_ / m) * np.concatenate([[0], weights[1:]])

    # calculate gradient 
    gradient = (1/m) * X.T @ (y_pred - y) + reg_term
    return weights - lr * gradient

# Training loop
def train_logistic_regression(X, y, lr=0.01, epochs=1000, lambda_=0.01):
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    weights = np.zeros(X.shape[1])
    losses = []

    for epoch in range(epochs):
        y_pred = sigmoid(X @ weights)
        loss = compute_loss(y, y_pred)
        losses.append(loss)
        weights = update_weights(X, y, weights, lr, lambda_)

        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    return weights, losses

# Predict function
def predict(X, weights):
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    probs = sigmoid(X @ weights)
    return (probs >= 0.5).astype(int)

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

    epochs = 1000
    lr = 0.05
    lambda_ = 0.1

    # Train/Validation/Test split (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(X_np, y_np, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train the model
    weights, losses = train_logistic_regression(X_train, y_train, lr, epochs, lambda_)

    # Validate the model
    y_val_pred = predict(X_val, weights)

    print("MAP")
    print(f"Epochs: {epochs}, Learning Rate: {lr}, Lambda: {lambda_}")

    # Print validation accuracy
    val_accuracy = np.mean(y_val_pred == y_val)
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
    plt.title('Training Loss over Epochs (MAP)')
    plt.plot(range(epochs), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('map_training_loss.jpg')

    # Draw a confusion matrix for the validation set
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_val, y_val_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Fail", "Pass"])
    disp.plot()
    plt.title("MAP Confusion Matrix")
    plt.savefig("map_confusion_matrix.jpg")


main()