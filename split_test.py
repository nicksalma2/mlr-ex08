"""Does last year's data improve generalisation to *unseen* other-phone photographs?

The main table cannot answer this for the combination model: its historical training images
and the historical validation set come from the same archive, so scoring it there leaks.
Here the historical training folder is split by a fixed rule (every third file of each
class goes to the test side), so the test images are from other phones and never trained on.

    python -I split_test.py --seed 0 [--out split_s0.csv]

D0 = pretrained network, our 101 photographs only          (one change: no historical data)
D1 = the same network and settings, plus the other 2/3 of the historical photographs
Both scored on the same 236 held-out historical images, at the best group-validation epoch.
"""

import argparse
import csv
import os
import sys
import time

import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_ablations as R

GROUP = r"C:\Users\Νικολας Σαλματανης\Desktop\MLR-ex08\ex08_leaves_group1\leaves"
HIST = r"C:\Users\Νικολας Σαλματανης\Desktop\MLR-ex08\ex08_leaves_images\leaves"


def thirds(folder, transform):
    """Every third file of each class (in sorted order) → the test side; the rest → training."""
    base = R.LeafDataset(os.path.join(folder, "train"), transform)
    test, train = R.LeafDataset(os.path.join(folder, "train"), transform), \
        R.LeafDataset(os.path.join(folder, "train"), transform)
    test.samples = [s for c in range(len(base.classes))
                    for j, s in enumerate([s for s in base.samples if s[1] == c]) if j % 3 == 0]
    train.samples = [s for c in range(len(base.classes))
                     for j, s in enumerate([s for s in base.samples if s[1] == c]) if j % 3 != 0]
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    aug = R.make_transform(jitter=True)
    hist_train_aug, hist_test_aug = thirds(HIST, aug)
    hist_test_plain = R.LeafDataset(os.path.join(HIST, "train"), R.transform)
    hist_test_plain.samples = hist_test_aug.samples        # the same files, clean preprocessing
    group_train = R.LeafDataset(os.path.join(GROUP, "train"), aug)
    group_val = R.LeafDataset(os.path.join(GROUP, "val"), R.transform)
    print(f"historical: {len(hist_train_aug)} for training, {len(hist_test_plain)} held out", flush=True)

    loader = lambda ds: DataLoader(ds, batch_size=32, shuffle=False)
    rows = []
    for name, train_ds in [("D0 group only", group_train),
                           ("D1 group + historical", ConcatDataset([group_train, hist_train_aug]))]:
        torch.manual_seed(args.seed)
        gen = torch.Generator().manual_seed(args.seed)
        model = R.mobilenet().to(R.device)
        t0 = time.time()
        log = []
        best_group = R.train_leaf(model, DataLoader(train_ds, batch_size=32, shuffle=True, generator=gen),
                                  train_ds, loader(group_val), group_val, epochs=40, cosine=True,
                                  keep_best=True, log=log)
        group_final, _, _ = R.evaluate(model, loader(group_val), len(group_val))
        slice_acc, preds, truths = R.evaluate(model, loader(hist_test_plain), len(hist_test_plain))
        balanced = sum(sum(p == t for p, t in zip(preds, truths) if t == c) / sum(truths == c)
                       for c in range(4)) / 4
        print(f"[{name} seed{args.seed}] group best {best_group:.4f} final {group_final:.4f} "
              f"| held-out historical {slice_acc:.4f} balanced {balanced:.4f} ({time.time() - t0:.0f}s)",
              flush=True)
        rows.append([name, args.seed, f"{best_group:.4f}", f"{group_final:.4f}",
                     f"{slice_acc:.4f}", f"{balanced:.4f}", f"{time.time() - t0:.0f}"])

    if args.out:
        new = not os.path.exists(args.out)
        with open(args.out, "w" if new else "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["variant", "seed", "group_best", "group_final", "held_out_hist",
                            "held_out_balanced", "seconds"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
