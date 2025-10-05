import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Load the test data (first 202 samples)
X_test_2 = pd.read_csv('X_test_2.csv').values[:202]
y_test_2 = pd.read_csv('y_test_2_reduced.csv').values[:202]

# Select the same top 200 features as used in training
X_test_2_selected = X_test_2[:, indices]

# Standardize test features using the same scaler
X_test_2_scaled = scaler.transform(X_test_2_selected)


# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_2_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_2, dtype=torch.long)

# Load the trained DNN model
model = DNNModel(input_dim=200, num_classes=28)
model.load_state_dict(torch.load('best_model_RD_1.pth'))
model.eval()


# Move model to appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare DataLoader for the test set
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Lists to store true labels and predictions
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Ensure all labels and predictions are integers
y_true = [int(x) for x in y_true]
y_pred = [int(x) for x in y_pred]


# Calculate and print confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)


# Generate and print classification report
unique_classes = sorted(list(set(y_true)))
report = classification_report(y_true, y_pred, target_names=[str(i) for i in unique_classes])
print("Classification Report:")
print(report)

# Compute and print weighted and unweighted F1 scores
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"F1 Score (Weighted): {f1:.4f}")

f1_unweighted = f1_score(y_true, y_pred, average='macro')
print(f"F1 Score (Unweighted): {f1_unweighted:.4f}")