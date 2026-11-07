"""Average the ablation runs over seeds and print the report table.

    python -I summarize.py [results2.csv]

Every accuracy is printed as a mean over seeds with the min-max spread, because with
three seeds a standard deviation suggests more precision than the runs contain.
"""

import csv
import sys
from collections import defaultdict

METRICS = ["final_val", "best_val", "balanced_val", "camera_like", "tta_val", "hist_holdout"]
LABELS = {
    "B0": "3c  LeafNet from scratch, no augmentation",
    "B1": "3d  LeafNetBN + geometric augmentation",
    "E1": "E1  + ColorJitter",
    "E4a": "E4a + dropout 0.3",
    "E4b": "E4b + global average pooling",
    "E4c": "E4c + AdamW, weight decay 1e-4",
    "E4d": "E4d + label smoothing 0.1",
    "E5": "E5  + 700 historical training images",
    "B2": "T2  MobileNet, whole network fine-tuned",
    "E2": "E2  MobileNet, backbone frozen",
    "E3": "E3  MobileNet + aug + cosine + 40 epochs",
    "BEST": "BEST  E1 + E3 + E5 combined",
}
ORDER = ["B0", "B1", "E1", "E4a", "E4b", "E4c", "E4d", "E5", "B2", "E2", "E3", "BEST"]
POINTS_PER_IMAGE = 100.0 / 28          # the group validation set has 28 images


def main():
    paths = sys.argv[1:] or ["results2.csv"]
    runs = defaultdict(dict)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                runs[row["variant"]][row["seed"]] = row

    print(f"{len(runs)} variants, seeds per variant: "
          f"{', '.join(f'{v}={len(runs[v])}' for v in ORDER if v in runs)}\n")

    header = f"| {'variant':40s} | {'final val':>16s} | {'best val':>16s} | {'balanced val':>16s} | " \
             f"{'camera-like':>16s} | {'TTA':>16s} | {'hist holdout':>16s} |"
    print(header)
    print("|" + ("-" * 42 + "|") * 7)

    means = {}
    for name in ORDER:
        if name not in runs:
            continue
        cells, means[name] = [], {}
        for metric in METRICS:
            values = [float(runs[name][s][metric]) for s in runs[name]
                      if runs[name][s][metric] != "nan"]
            if not values:
                cells.append("   leak   (n/a) ")
                continue
            mean = sum(values) / len(values)
            means[name][metric] = mean
            cells.append(f" {mean:.3f} [{min(values):.2f}-{max(values):.2f}]")
        print(f"| {LABELS[name]:40s} | {' | '.join(f'{c:>16s}' for c in cells)} |")

    print(f"\none validation image = {POINTS_PER_IMAGE:.1f} accuracy points (28 images)")
    print("one historical image = %.1f points (120 images)\n" % (100.0 / 120))

    for base in ("B1", "B2"):
        if base not in means:
            continue
        print(f"change vs {base} ({LABELS[base]}), mean over seeds:")
        for name in ORDER:
            if name == base or name not in means:
                continue
            parts = []
            for metric in ("final_val", "best_val", "camera_like", "hist_holdout"):
                if metric in means[name] and metric in means[base]:
                    delta = 100 * (means[name][metric] - means[base][metric])
                    parts.append(f"{metric.replace('_val', '')} {delta:+5.1f} ({delta / POINTS_PER_IMAGE:+4.1f} img)")
            print(f"  {LABELS[name]:42s} " + "  ".join(parts))
        print()


if __name__ == "__main__":
    main()
