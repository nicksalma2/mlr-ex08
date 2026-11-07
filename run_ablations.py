"""Task 4 ablations, run locally on the two datasets already on disk.

The dataset, network and training-loop code below is the notebook code verbatim
(Task 3 LeafDataset/LeafNet/LeafNetBN + the 3c loop as train_leaf), so the numbers
here come from the same code path as the cells handed over for the report.

    python -I run_ablations.py --group <...>/leaves --hist <...>/leaves [--seeds 0 1 2] [--only B0 E1 ...]
"""

import argparse
import copy
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import models, transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LeafDataset(Dataset):
    def __init__(self, folder, transform):
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_folder = os.path.join(folder, class_name)
            for file_name in sorted(os.listdir(class_folder)):
                file_path = os.path.join(class_folder, file_name)
                if os.path.isfile(file_path):
                    self.samples.append((file_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class LeafNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeafNetBN(nn.Module):
    def __init__(self, pool="flatten", dropout=0.0):        # pool="gap" -> global average pooling
        super().__init__()
        self.gap = pool == "gap"
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 if self.gap else 64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.mean([2, 3]) if self.gap else x.flatten(1)
        x = self.drop(torch.relu(self.fc1(x)))
        return self.fc2(x)


def evaluate(model, loader, n, seed=None):
    if seed is not None:
        torch.manual_seed(0)
    model.eval()
    correct, preds, truths = 0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            preds.append(predictions.cpu())
            truths.append(labels.cpu())
    return correct / n, torch.cat(preds).numpy(), torch.cat(truths).numpy()


def train_leaf(model, train_loader, train_ds, val_loader, val_ds,
               epochs=15, lr=1e-3, adamw=False, weight_decay=0.0,
               label_smoothing=0.0, keep_best=False, cosine=False, log=None):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay) if adamw \
        else torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if cosine else None

    best_acc, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(labels)

        val_accuracy = evaluate(model, val_loader, len(val_ds))[0]
        if val_accuracy > best_acc:
            best_acc, best_state = val_accuracy, copy.deepcopy(model.state_dict())
        line = (f"Epoch {epoch + 1:2d}/{epochs}: Train Loss = {total_train_loss / len(train_ds):.4f}, "
                f"Val Acc = {val_accuracy:.4f}")
        print(line, flush=True)
        if log is not None:
            log.append(line)
        if scheduler is not None:
            scheduler.step()

    if keep_best:
        model.load_state_dict(best_state)
    return best_acc


# ---------------------------------------------------------------- the variants

def make_transform(jitter=False):
    jitter_list = [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)] if jitter else []
    return transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
    ] + jitter_list + [transforms.ToTensor(), transforms.Normalize(mean, std)])


cam_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.5, 1.0)),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.15),
    transforms.ToTensor(), transforms.Normalize(mean, std)])


def mobilenet(freeze=False):
    model = models.mobilenet_v3_small(weights="DEFAULT")
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    return model.to(device)


def build(name, D, seed):
    """Build + train one variant. Returns (model, best_acc, log)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    log = []
    plain_loader = lambda ds: DataLoader(ds, batch_size=32, shuffle=True, generator=gen)

    if name == "B0":                      # Task 3c: LeafNet from scratch, no augmentation
        ds = D["group_train_plain"]
        model = LeafNet().to(device)
        train_leaf(model, plain_loader(ds), ds, D["val"], D["group_val"], log=log)
    elif name == "B1":                    # Task 3d: LeafNetBN + geometric augmentation
        ds = D["group_train_aug"]
        model = LeafNetBN().to(device)
        train_leaf(model, plain_loader(ds), ds, D["val"], D["group_val"], log=log)
    elif name == "E1":                    # B1 + ColorJitter
        ds = LeafDataset(os.path.join(D["group"], "train"), make_transform(jitter=True))
        model = LeafNetBN().to(device)
        train_leaf(model, plain_loader(ds), ds, D["val"], D["group_val"], log=log)
    elif name.startswith("E4"):           # B1 + one regularizer / classifier change
        kw = {"E4a": {"dropout": 0.3}, "E4b": {"pool": "gap"},
              "E4c": None, "E4d": None}[name]
        ds = D["group_train_aug"]
        model = LeafNetBN(**(kw or {})).to(device)
        extra = {"E4c": {"adamw": True, "weight_decay": 1e-4}, "E4d": {"label_smoothing": 0.1}}.get(name, {})
        train_leaf(model, plain_loader(ds), ds, D["val"], D["group_val"], log=log, **extra)
    elif name == "E5":                    # B1 + 700 historical training images
        ds = ConcatDataset([D["group_train_aug"], D["hist_train_aug"]])
        model = LeafNetBN().to(device)
        train_leaf(model, plain_loader(ds), ds, D["val"], D["group_val"], log=log)
    elif name == "B2":                    # Task 2: whole MobileNet fine-tuned, 15 epochs, no augmentation
        ds = D["group_train_plain"]
        model = mobilenet().to(device)
        train_leaf(model, plain_loader(ds), ds, D["val"], D["group_val"], log=log)
    elif name == "E2":                    # same, backbone frozen, only the new last layer trained
        ds = D["group_train_plain"]
        model = mobilenet(freeze=True).to(device)
        train_leaf(model, plain_loader(ds), ds, D["val"], D["group_val"], log=log)
    elif name == "E3":                    # MobileNet + augmentation + cosine schedule + 40 epochs
        ds = D["group_train_aug"]
        model = mobilenet().to(device)
        train_leaf(model, plain_loader(ds), ds, D["val"], D["group_val"], epochs=40, cosine=True, log=log)
    elif name == "BEST":                  # the winners combined: E1 + E3 + E5
        aug = make_transform(jitter=True)
        ds = ConcatDataset([LeafDataset(os.path.join(D["group"], "train"), aug),
                            LeafDataset(os.path.join(D["hist"], "train"), aug)])
        model = mobilenet().to(device)
        train_leaf(model, plain_loader(ds), ds, D["val"], D["group_val"], epochs=40, cosine=True, log=log)
    else:
        raise KeyError(name)
    return model, log


def predict_meaned(model, plain_ds, cam_ds, copies=4):
    # the logits of the clean image and of `copies` camera copies, averaged before the argmax
    with torch.no_grad():
        logits = None
        for ds, seed in [(plain_ds, None)] + [(cam_ds, s) for s in range(1, copies + 1)]:
            if seed is not None:
                torch.manual_seed(seed)
            out = torch.cat([model(images.to(device))
                             for images, _ in DataLoader(ds, batch_size=32, shuffle=False)])
            logits = out if logits is None else logits + out
    return (logits / (copies + 1)).argmax(dim=1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--hist", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    D = {
        "group": args.group,
        "hist": args.hist,
        "group_train_plain": LeafDataset(os.path.join(args.group, "train"), transform),
        "group_train_aug": LeafDataset(os.path.join(args.group, "train"), make_transform()),
        "group_val": LeafDataset(os.path.join(args.group, "val"), transform),
        "hist_train_aug": LeafDataset(os.path.join(args.hist, "train"), make_transform()),
        "hist_val": LeafDataset(os.path.join(args.hist, "val"), transform),
    }
    for key in ("group_train_plain", "group_train_aug", "group_val", "hist_train_aug", "hist_val"):
        assert D[key].classes == ["apple", "cherry", "chestnut", "maple"] and len(D[key]) > 0, \
            f"{key}: {D[key].classes} with {len(D[key])} images"
    D["cam_ds"] = LeafDataset(os.path.join(args.group, "val"), cam_transform)
    D["val"] = DataLoader(D["group_val"], batch_size=32, shuffle=False)
    for key, ds in (("cam", D["cam_ds"]), ("hist_val", D["hist_val"])):
        D[key] = DataLoader(ds, batch_size=32, shuffle=False)
        D[key + "_n"] = len(ds)
    print("group train", len(D["group_train_plain"]), "| val", len(D["group_val"]),
          "| hist train", len(D["hist_train_aug"]), "| hist val", len(D["hist_val"]), flush=True)

    names = args.only or ["B0", "B1", "E1", "E4a", "E4b", "E4c", "E4d", "E5", "B2", "E2", "E3"]
    new_file = not os.path.exists(args.out)
    out = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.writer(out)
    if new_file:
        writer.writerow(["variant", "seed", "final_val", "best_val", "balanced_val",
                         "camera_like", "tta_val", "hist_holdout", "hist_holdout_balanced", "seconds"])
    os.makedirs("weights", exist_ok=True)

    for name in names:
        for seed in args.seeds:
            t0 = time.time()
            model, log = build(name, D, seed)
            final_val, preds, truths = evaluate(model, D["val"], len(D["group_val"]))
            best_val = max(float(l.split("Val Acc = ")[1]) for l in log)
            balanced = float(np.mean([np.mean(preds[truths == i] == i) for i in range(4)]))
            cam_acc, _, _ = evaluate(model, D["cam"], D["cam_n"], seed=0)
            tta = float(np.mean(predict_meaned(model, D["group_val"], D["cam_ds"]) == truths))
            torch.save(model.state_dict(), f"weights/{name}_s{seed}.pt")
            if name in ("E5", "BEST"):
                hist_acc = hist_bal = float("nan")      # leakage: the historical train set was trained on
            else:
                hist_acc, hp, ht = evaluate(model, D["hist_val"], D["hist_val_n"])
                hist_bal = float(np.mean([np.mean(hp[ht == i] == i) for i in range(4)]))
            secs = time.time() - t0
            writer.writerow([name, seed, f"{final_val:.4f}", f"{best_val:.4f}", f"{balanced:.4f}",
                             f"{cam_acc:.4f}", f"{tta:.4f}", f"{hist_acc:.4f}", f"{hist_bal:.4f}", f"{secs:.0f}"])
            out.flush()
            with open("epochs.json", "a", encoding="utf-8") as f:
                json.dump({"variant": name, "seed": seed, "log": log}, f)
                f.write("\n")
            print(f"[{name} seed{seed}] final {final_val:.4f} best {best_val:.4f} bal {balanced:.4f} "
                  f"cam {cam_acc:.4f} hist {hist_acc:.4f} ({secs:.0f}s)", flush=True)
    out.close()


if __name__ == "__main__":
    main()
