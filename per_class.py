"""Per-species accuracy and confusion counts for the saved weights (no retraining).

Plain accuracy on 28 images hides which species you lost, and the classes are uneven
(4 apple / 10 cherry / 6 chestnut / 8 maple), so one wrong apple costs 25 points.

    python -I per_class.py
"""

import glob
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_ablations as R
from tta_check import GROUP, make_model

HIST_DIR = r"C:\Users\Νικολας Σαλματανης\Desktop\MLR-ex08\ex08_leaves_images\leaves"


def accuracy_per_class(model, loader, n_classes):
    correct = np.zeros(n_classes)
    total = np.zeros(n_classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)      # rows true, columns predicted
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(R.device)).argmax(dim=1).cpu()
            for truth, pred in zip(labels.numpy(), predictions.numpy()):
                total[truth] += 1
                correct[truth] += truth == pred
                matrix[truth][pred] += 1
    return correct / total, total, matrix


def main():
    classes = ["apple", "cherry", "chestnut", "maple"]
    sets = {
        "clean val": R.LeafDataset(os.path.join(GROUP, "val"), R.transform),
        "camera-like": R.LeafDataset(os.path.join(GROUP, "val"), R.cam_transform),
        "hist holdout": R.LeafDataset(os.path.join(HIST_DIR, "val"), R.transform),
    }

    per_variant = defaultdict(lambda: defaultdict(list))
    matrices = {}
    for path in sorted(glob.glob("weights/*_s[012].pt")):
        name = os.path.basename(path).rsplit("_s", 1)[0]
        model = make_model(name).to(R.device)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        for label, ds in sets.items():
            if name in ("E5", "BEST") and label == "hist holdout":
                continue                                   # leakage: those saw the historical train set
            acc, _, matrix = accuracy_per_class(model, DataLoader(ds, batch_size=32, shuffle=False), 4)
            per_variant[name][label].append(acc)
            if label == "hist holdout" and path.endswith("_s0.pt"):
                matrices[name] = matrix

    for label in sets:
        labels = [y for _, y in sets[label].samples]
        print(f"\n=== {label}  ({', '.join(f'{c}={labels.count(i)}' for i, c in enumerate(classes))}) ===")
        print(f"{'variant':6s} " + " ".join(f"{c:>10s}" for c in classes) + f" {'balanced':>10s}")
        for name in ["B0", "B1", "E1", "E4b", "E4a", "E4c", "E4d", "E5", "B2", "E2", "E3", "BEST"]:
            if label not in per_variant[name]:
                continue
            mean = np.mean(np.array(per_variant[name][label]), axis=0)
            print(f"{name:6s} " + " ".join(f"{v:>10.2f}" for v in mean) + f" {mean.mean():>10.3f}")

    print("\n=== confusion on the historical holdout (seed 0; rows true, columns predicted) ===")
    for name, matrix in matrices.items():
        print(f"\n{name}")
        print("            " + " ".join(f"{c:>9s}" for c in classes))
        for i, row in enumerate(matrix):
            print(f"  {classes[i]:9s}" + " ".join(f"{v:>9d}" for v in row))


if __name__ == "__main__":
    main()
