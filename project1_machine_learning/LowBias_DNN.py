import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Load training data
X_train = pd.read_csv('X_train.csv').values
Y_train = pd.read_csv('y_train.csv').values

# Shuffle the training data
X_train, Y_train = shuffle(X_train, Y_train, random_state=42)


# Train a Random Forest to get feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)

# Select top 200 important features
importances = rf.feature_importances_
indices = importances.argsort()[::-1][:200]
X_train_selected = X_train[:, indices]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_train_final, X_val_final, Y_train_final, Y_val_final = train_test_split(X_train_scaled, Y_train, test_size=0.2,
                                                                          random_state=42)
# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_final, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_final, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_final, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val_final, dtype=torch.long)

# Squeeze tensors to remove extra dimensions if necessary
Y_train_tensor = Y_train_tensor.squeeze()
Y_val_tensor = Y_val_tensor.squeeze()

# Define custom Focal Loss function
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=28, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)

        pt = torch.exp(-CE_loss)
        F_loss = (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# Define DNN model architecture
class DNNModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        super(DNNModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

# Model and training setup
input_dim = 200
num_classes = 28
model = DNNModel(input_dim=input_dim, num_classes=num_classes)

batch_size = 32
epochs = 100
learning_rate = 3e-5

# Prepare DataLoaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define loss function and optimizer
criterion = FocalLoss(alpha=0.25, gamma=2, num_classes=num_classes, reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Learning rate scheduler: reduce LR if validation accuracy stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=15, verbose=True)

# Early stopping parameters
best_val_accuracy = 0.0
patience = 15
counter = 0

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_preds / total_preds
    print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}')

    model.eval()
    val_loss = 0.0
    val_correct_preds = 0
    val_total_preds = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_correct_preds += (predicted == labels).sum().item()
            val_total_preds += labels.size(0)

    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_accuracy = val_correct_preds / val_total_preds
    print(f'Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}')

     # Save best model based on validation accuracy
    if epoch_accuracy >= 0.8:
        if val_epoch_accuracy > best_val_accuracy:
            best_val_accuracy = val_epoch_accuracy

            torch.save(model.state_dict(), 'best_model_RD.pth')
            counter = 0  # reset counter
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    scheduler.step(val_epoch_accuracy)

# Final evaluation on validation set
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())



# Print confusion matrix and classification report
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

unique_classes = sorted(list(set(y_true)))
report = classification_report(y_true, y_pred, target_names=[str(i) for i in unique_classes])
print("Classification Report:")
print(report)

# Calculate F1 scores
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"F1 Score (Weighted): {f1:.4f}")

f1_unweighted = f1_score(y_true, y_pred, average='macro')
print(f"F1 Score (Unweighted): {f1_unweighted:.4f}")

# Plot the top 20 feature importances
top_20_indices = importances.argsort()[::-1][:20]
top_20_importances = importances[top_20_indices]

plt.figure(figsize=(10, 6))
plt.barh(range(20), top_20_importances, align="center")
plt.yticks(range(20), top_20_indices)
plt.xlabel("Feature Importance")
plt.title("Top 20 Feature Importances")
plt.show()