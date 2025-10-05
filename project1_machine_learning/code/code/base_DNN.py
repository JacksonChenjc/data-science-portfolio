#baseline DNN
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch.utils.data import TensorDataset, DataLoader

# 1. Load the data
X = pd.read_csv('X_train.csv').values
y = pd.read_csv('y_train.csv').values.squeeze()

# 2. Standardize the features (zero mean and unit variance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split the dataset into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Convert datasets to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# 5. Wrap the datasets into DataLoaders for batch processing
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32, shuffle=False)

# 6. Define a basic fully connected DNN architecture
class BasicDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BasicDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = BasicDNN(input_dim=X_train.shape[1], num_classes=28)

# Instantiate the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# 8. Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 9. Train the model for a fixed number of epochs (100)
epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print(f'Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss/len(train_loader):.4f} Accuracy: {correct/total:.4f}')

# 10. Evaluate the model on the validation set
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

cm = confusion_matrix(y_true, y_pred)
# Calculate confusion matrix and classification metrics
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n", classification_report(y_true, y_pred))
print(f"\nWeighted F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
print(f"\n unweighted F1 Score: {f1_score(y_true, y_pred, average='macro'):.4f}")


