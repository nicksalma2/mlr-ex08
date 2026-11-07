"""Train the ship candidate once with keep_best, export it to ONNX, and prove the exported
file answers like PyTorch.

The weights saved by run_ablations.py are the *last* epoch (train_leaf was called without
keep_best), while the report quotes the best epoch — so this retrains seed 0 and keeps the
best weights before exporting. Writes only into exports/; nothing in the repository is touched.
"""

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_ablations as R

GROUP = r"C:\Users\Νικολας Σαλματανης\Desktop\MLR-ex08\ex08_leaves_group1\leaves"
HIST = r"C:\Users\Νικολας Σαλματανης\Desktop\MLR-ex08\ex08_leaves_images\leaves"
CLASSES = ["apple", "cherry", "chestnut", "maple"]
SEED = 0


def main():
    # the ONNX exporter prints a ✅, which the Greek Windows console (cp1253) cannot encode
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.makedirs("exports", exist_ok=True)
    aug = R.make_transform(jitter=True)
    train_ds = ConcatDataset([R.LeafDataset(os.path.join(GROUP, "train"), aug),
                              R.LeafDataset(os.path.join(HIST, "train"), aug)])
    val_ds = R.LeafDataset(os.path.join(GROUP, "val"), R.transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    model = R.mobilenet().to(R.device)
    checkpoint = "exports/best_checkpoint.pt"
    if os.path.exists(checkpoint):            # trained already: reuse the best-epoch weights
        saved = torch.load(checkpoint, weights_only=True)
        model.load_state_dict(saved["state"])
        best_acc = saved["best"]
        print("reusing the saved best-epoch weights", flush=True)
    else:
        best_acc = R.train_leaf(model, DataLoader(train_ds, batch_size=32, shuffle=True, generator=gen),
                                train_ds, val_loader, val_ds, epochs=40, cosine=True, keep_best=True)
        torch.save({"state": model.state_dict(), "best": best_acc}, checkpoint)
    final_acc, preds, truths = R.evaluate(model, val_loader, len(val_ds))
    camera, _, _ = R.evaluate(model, DataLoader(
        R.LeafDataset(os.path.join(GROUP, "val"), R.cam_transform), batch_size=32, shuffle=False),
        len(val_ds), seed=0)
    print(f"exported model: group val best {best_acc:.4f} final {final_acc:.4f} camera-like {camera:.4f}",
          flush=True)

    model.eval().cpu()
    torch.onnx.export(model, (torch.zeros(1, 3, 96, 96),), "exports/leaf_model_best.onnx",
                      input_names=["image"], output_names=["logits"], external_data=False)

    import onnx
    model_info = {"classes": CLASSES, "input_size": 96, "mean": R.mean, "std": R.std}
    onnx_model = onnx.load("exports/leaf_model_best.onnx")
    onnx_model.metadata_props.add(key="model_info", value=json.dumps(model_info))
    onnx.save(onnx_model, "exports/leaf_model_best.onnx")
    with open("exports/model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f)
    print(f"exports/leaf_model_best.onnx: "
          f"{os.path.getsize('exports/leaf_model_best.onnx') / 1e6:.1f} MB", flush=True)

    import onnxruntime
    session = onnxruntime.InferenceSession("exports/leaf_model_best.onnx",
                                           providers=["CPUExecutionProvider"])
    print("onnx input:", [(i.name, i.shape) for i in session.get_inputs()],
          "| output:", [o.name for o in session.get_outputs()])

    with torch.no_grad():
        torch_logits = torch.cat([model(images.to(R.device)) for images, _ in val_loader]).numpy()
    # the exported file takes one frame at a time, exactly as cell 49 exports it and the page feeds it
    onnx_logits = np.concatenate([session.run(["logits"], {"image": images.numpy()})[0]
                                  for images, _ in DataLoader(val_ds, batch_size=1, shuffle=False)])
    onnx_preds = onnx_logits.argmax(axis=1)
    print(f"worst |torch - onnx| logit difference: {np.abs(torch_logits - onnx_logits).max():.2e}")
    print(f"same prediction on all {len(val_ds)} validation images: {bool((onnx_preds == preds).all())}")
    print(f"onnx accuracy on the group validation set: {(onnx_preds == truths).mean():.4f}")


if __name__ == "__main__":
    main()
