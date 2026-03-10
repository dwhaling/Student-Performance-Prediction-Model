import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

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

    # Train/Validation/Test split (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(X_np, y_np, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train naive bayes model
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_val_pred = gnb.predict(X_val)

    print("Naive Bayes Model")
    # print(f"Epochs: {epochs}, Learning Rate: {lr}")

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

    # Draw a confusion matrix for the validation set
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_val, y_val_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Fail", "Pass"])
    disp.plot()
    plt.title("Naive Bayes Confusion Matrix")
    plt.savefig("naive_bayes_confusion_matrix.jpg")


main()