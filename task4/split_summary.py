"""Average the held-out-split runs (does last year's data help on unseen other-phone images?)."""

import csv

rows = []
for seed in (0, 1, 2):
    with open(f"split_s{seed}.csv", encoding="utf-8") as f:
        rows += list(csv.DictReader(f))

values = {}
for name in ("D0 group only", "D1 group + historical"):
    selected = [r for r in rows if r["variant"] == name]
    hold = [float(r["held_out_hist"]) for r in selected]
    group = [float(r["group_best"]) for r in selected]
    values[name] = hold
    print(f"{name:22s} held-out historical mean {sum(hold) / len(hold):.4f} "
          f"[{min(hold):.3f}-{max(hold):.3f}]  group val mean {sum(group) / len(group):.4f}  "
          f"| per seed " + " ".join(f"{h:.3f}" for h in hold))

gain = [100 * (b - a) for a, b in zip(values["D0 group only"], values["D1 group + historical"])]
print(f"\ngain from the 464 historical images on the 236 held-out images, per seed: "
      + "  ".join(f"{g:+.1f} pts" for g in gain)
      + f"   (mean {sum(gain) / len(gain):+.1f} = {sum(gain) / len(gain) / (100 / 236):+.0f} images)")
