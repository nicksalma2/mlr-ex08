"""Re-examine test-time averaging on the saved weights (no retraining).

The sweep averaged *logits* over the clean image and four aggressively corrupted camera
copies. This separates the two possible reasons that could hurt: the aggregation rule
(sum of logits follows the most confident view, mean of softmax follows the majority) and
the choice of ensemble members (rotation 30 + colour jitter 0.4 is a corruption, not a
nuisance variation). Mild copies = small crops only.

    python -I tta_check.py
"""

import glob
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I leaves no local path
import run_ablations as R

GROUP = r"C:\Users\Νικολας Σαλματανης\Desktop\MLR-ex08\ex08_leaves_group1\leaves"
device = R.device

mild_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
    transforms.ToTensor(), transforms.Normalize(R.mean, R.std)])


def make_model(name):
    if name == "B0":
        return R.LeafNet()
    if name == "E4a":
        return R.LeafNetBN(dropout=0.3)
    if name == "E4b":
        return R.LeafNetBN(pool="gap")
    if name == "E2":
        return R.mobilenet(freeze=True)
    if name in ("B2", "E3", "BEST"):
        return R.mobilenet()
    return R.LeafNetBN()


def logits_over(model, ds, seeds):
    """Sum the logits over one pass of ds per seed (seed None = nothing to seed)."""
    total = None
    for seed in seeds:
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            out = torch.cat([model(images.to(device))
                             for images, _ in DataLoader(ds, batch_size=32, shuffle=False)])
        total = out if total is None else total + out
    return total


def main():
    plain = R.LeafDataset(os.path.join(GROUP, "val"), R.transform)
    strong = R.LeafDataset(os.path.join(GROUP, "val"), R.cam_transform)
    mild = R.LeafDataset(os.path.join(GROUP, "val"), mild_transform)
    truths = np.array([label for _, label in plain])

    rows = defaultdict(list)
    for path in sorted(glob.glob("weights/*_s[012].pt")):
        name = os.path.basename(path).rsplit("_s", 1)[0]
        model = make_model(name).to(device)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()

        clean = logits_over(model, plain, [None])
        cam = logits_over(model, strong, [1, 2, 3, 4])
        milds = logits_over(model, mild, [1, 2, 3, 4])
        p_clean, p_cam, p_mild = (torch.softmax(x, dim=1) for x in (clean, cam / 5, milds / 5))

        rows[name].append([
            (clean.argmax(1).numpy() == truths).mean(),                    # clean, single pass
            ((clean + cam).argmax(1).numpy() == truths).mean(),            # what the sweep ran
            ((p_clean + p_cam).argmax(1).numpy() == truths).mean(),        # probs, clean + 4 corrupt
            ((p_clean + p_mild).argmax(1).numpy() == truths).mean(),       # probs, clean + 4 mild
            (p_cam.argmax(1).numpy() == truths).mean(),                    # 4 corrupt copies only
        ])

    cols = ["clean", "sum logits +4 strong", "mean prob +4 strong", "mean prob +4 mild",
            "4 strong only"]
    print(f"mean over 3 seeds, accuracy on the {len(truths)} group validation images\n")
    print(f"{'variant':6s} " + " ".join(f"{c:>20s}" for c in cols))
    for name in ["B0", "B1", "E1", "E4a", "E4b", "E4c", "E4d", "E5", "B2", "E2", "E3", "BEST"]:
        if name in rows:
            values = np.mean(np.array(rows[name]), axis=0)
            print(f"{name:6s} " + " ".join(f"{v:>20.3f}" for v in values))


if __name__ == "__main__":
    main()
