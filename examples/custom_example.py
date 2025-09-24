import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import KMNIST
from torchvision.models import resnet18
from tqdm import tqdm

from sslpack.algorithms import FixMatch
from sslpack.utils.data import BasicDataset, CyclicLoader, TransformDataset, split_lbl_ulbl
# from sslpack.utils.dist_align import DistributionAlignment
from sslpack.algorithms.utils import DistributionAlignment

#%% Load dataset
train = KMNIST(root="~/.sslpack/datasets", train=True, download=True)
test = KMNIST(root="~/.sslpack/datasets", train=False, download=True)
num_classes = len(train.classes)

X_tr, y_tr = train.data.unsqueeze(1)/255, train.targets
X_ts, y_ts = test.data.unsqueeze(1)/255, test.targets

#%% Define transforms
crop_size = 28
crop_ratio = 0.875  # 24.5/28

transform_weak = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(crop_size),
    transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_strong = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(crop_size),
    transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(3, 5),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(crop_size),
    transforms.ToTensor(),
])

#%% sslpack dataset objects
num_labels_per_class = 10
X_tr_lb, y_tr_lb, X_tr_ulb, y_tr_ulb = split_lbl_ulbl(X_tr, y_tr, num_labels_per_class)

dataset_lb = TransformDataset(X_tr_lb, y_tr_lb, weak_transform=transform_weak, strong_transform=transform_strong)
dataset_ulb = TransformDataset(X_tr_ulb, y_tr_ulb, weak_transform=transform_weak, strong_transform=transform_strong)
dataset_val = TransformDataset(X_ts, y_ts, weak_transform=transform_val, strong_transform=transform_val)

batch_size_lb = 64
uratio = 7
batch_size_eval = 256
batch_size_ulb = uratio*batch_size_lb
train_loader = CyclicLoader(dataset_lb, dataset_ulb, lbl_batch_size=batch_size_lb, ulbl_batch_size=batch_size_ulb)
test_loader = DataLoader(dataset_val, batch_size=batch_size_eval)

#%% model
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x, only_features=False, only_classifier=False):
        if only_classifier:
            return self.classifier(x)

        x = self.features(x)
        if only_features:
            return x

        return self.classifier(x)

# model = CNN()
model = resnet18(num_classes=num_classes)
model._modules["conv1"] = nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3, bias=False)

#%% optimizer
lr = 0.03
num_train_iters = 1000
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: np.cos((7*np.pi*t)/(16*num_train_iters)))
device = "cuda" if torch.cuda.is_available() else "cpu"

#%% train the model
def dict_to_device(d, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in d.items()}

def train(model, train_loader, algorithm, optimizer, scheduler=None, num_iters=128, num_log_iters=10, device="cpu"):
    model.to(device)
    model.train()

    training_bar = tqdm(train_loader, total=num_iters, desc="Training", leave=True)

    for i, (batch_lb, batch_ulb) in enumerate(training_bar):
        batch_lb = dict_to_device(batch_lb, device)
        batch_ulb = dict_to_device(batch_ulb, device)

        optimizer.zero_grad()
        loss = algorithm.forward(model, batch_lb, batch_ulb)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if i % num_log_iters == 0:
            training_bar.set_postfix(loss = round(loss.item(), 4))
            print(f"Iteration: {i}, Total Loss: {loss.item():.6f}")

        if i > num_iters:
            break

# algorithm = FixMatch()
algorithm = FixMatch(dist_align=DistributionAlignment())

train(
    model=model,
    train_loader=train_loader,
    algorithm=algorithm,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    num_iters=num_train_iters,
    num_log_iters=int(num_train_iters/10),
)

#%% evaluate the model
def evaluate(model, eval_loader, device="cpu", sig_fig=4):
    model.to(device)
    model.eval()
    total_num = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in eval_loader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            num_batch = y.shape[0]
            total_num += num_batch
            logits = model(X)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        acc = accuracy_score(y_true, y_pred)
        print("accuracy: ", acc)
        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')

        with np.printoptions(suppress=True, precision=sig_fig):
            print('confusion matrix:\n' + np.array_str(cf_mat))
        model.train()

evaluate(model, test_loader, device=device)