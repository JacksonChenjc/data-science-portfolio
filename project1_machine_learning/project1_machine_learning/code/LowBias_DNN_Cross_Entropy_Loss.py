import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import shuffle
import numpy as np

# Define function to compute weighted log loss
def weighted_log_loss(y_true_onehot, y_pred_prob):
    class_counts = np.sum(y_true_onehot, axis=0)
    mask = class_counts > 0
    class_weights = np.zeros_like(class_counts)
    class_weights[mask] = 1.0 / class_counts[mask]
    class_weights /= np.sum(class_weights)
    sample_weights = np.dot(y_true_onehot, class_weights)
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred_prob, eps, 1 - eps)
    per_sample_loss = -np.sum(y_true_onehot * np.log(y_pred_clipped), axis=1)
    return np.mean(sample_weights * per_sample_loss)
# Load and shuffle the training data
X = pd.read_csv('X_train.csv').values
y = pd.read_csv('y_train.csv').values
X, y = shuffle(X, y, random_state=42)

# Feature selection using Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
indices = importances.argsort()[::-1][:200]
X_selected = X[:, indices]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long).squeeze()
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long).squeeze()


# Define Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        CE = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE)
        FL = (1-pt)**self.gamma * CE
        return torch.mean(FL) if self.reduction=='mean' else torch.sum(FL)

# Define Deep Neural Network (DNN) model
class DNN(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.l1 = nn.Linear(d,512)
        self.l2 = nn.Linear(512,512)
        self.l3 = nn.Linear(512,512)
        self.l4 = nn.Linear(512,c)
        self.r = nn.ReLU()
    def forward(self, x):
        x = self.r(self.l1(x))
        x = self.r(self.l2(x))
        x = self.r(self.l3(x))
        return self.l4(x)


# Initialize model, optimizer, and loss function
d=200; c=28; b=32; lr=3e-5; e=100
model = DNN(d,c)
opt = optim.Adam(model.parameters(), lr=lr)
crit = FocalLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare DataLoaders
tr = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=b, shuffle=True)
vl = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=b, shuffle=False)

# Learning rate scheduler and early stopping setup
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=15, verbose=True)
best=0; cnt=0

# Training loop
for _ in range(e):
    model.train()
    for xi, yi in tr:
        xi, yi = xi.to(device), yi.to(device)
        out = model(xi)
        loss = crit(out, yi)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    vt, vp, vp_prob = [], [], []
    with torch.no_grad():
        for xi, yi in vl:
            xi, yi = xi.to(device), yi.to(device)
            out = model(xi)
            prob = torch.softmax(out,1)
            vp_prob.append(prob.cpu().numpy())
            _, pr = torch.max(out,1)
            vp.extend(pr.cpu().numpy()); vt.extend(yi.cpu().numpy())
    vp_prob = np.vstack(vp_prob)
    yv_oh = np.zeros_like(vp_prob)
    for i, lbl in enumerate(vt): yv_oh[i,int(lbl)]=1
    acc = np.mean(np.array(vp)==np.array(vt))
    if acc>best:
        best=acc; cnt=0; torch.save(model.state_dict(),'best.pth')
    else:
        cnt+=1
        if cnt>=15: break
    sch.step(acc)

# Final evaluation on validation set
model.eval()
vt, vp, vp_prob = [], [], []
with torch.no_grad():
    for xi, yi in vl:
        xi = xi.to(device)
        out = model(xi)
        prob = torch.softmax(out,1)
        vp_prob.append(prob.cpu().numpy())
        _, pr = torch.max(out,1)
        vp.extend(pr.cpu().numpy()); vt.extend(yi.numpy())
vp_prob = np.vstack(vp_prob)
yv_oh = np.zeros_like(vp_prob)
for i, lbl in enumerate(vt): yv_oh[i,int(lbl)]=1
val_wll = weighted_log_loss(yv_oh, vp_prob)
print("Weighted Log Loss (Validation Set):", round(val_wll,4))
print(confusion_matrix(vt, vp))
print(classification_report(vt, vp))

# Load test set (first 202 samples)
test_X = pd.read_csv('X_test_2.csv').values[:202]
test_y = pd.read_csv('y_test_2_reduced.csv').values[:202].ravel()

# Apply same feature selection and scaling
test_sel = test_X[:, indices]
test_std = scaler.transform(test_sel)

# Test set evaluation
tp, tp_prob = [], []
with torch.no_grad():
    for i in range(0, len(test_std), b):
        batch = torch.tensor(test_std[i:i+b], dtype=torch.float32).to(device)
        out = model(batch)
        prob = torch.softmax(out,1)
        tp_prob.append(prob.cpu().numpy())
        _, pr = torch.max(out,1)
        tp.extend(pr.cpu().numpy())
tp_prob = np.vstack(tp_prob)
ty_oh = np.zeros_like(tp_prob)
for i, lbl in enumerate(test_y): ty_oh[i,int(lbl)]=1
test_wll = weighted_log_loss(ty_oh, tp_prob)
print("Weighted Log Loss (Test Set):", round(test_wll,4))
print(confusion_matrix(test_y, tp))
print(classification_report(test_y, tp))


