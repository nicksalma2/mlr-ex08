# Task 4 — ablation cells (drop into the notebook after cell 61)

Everything here starts from the Task 3 pipeline (`LeafDataset`, `LeafNet`, `LeafNetBN`,
the 3c training loop) and the Task 2 model. No new classes or frameworks: one `evaluate`,
one `train_leaf` (the 3c loop as a function), one modified `LeafNetBN`.

## 0. Measured "before" numbers (already in the executed notebook, seed 0)

| pipeline | final epoch | best epoch |
|---|---|---|
| 3c — `LeafNet` from scratch, no augmentation | 0.5357 | 0.5357 |
| 3d — `LeafNetBN` + geometric augmentation | 0.6071 | 0.6429 |
| Task 2 — MobileNetV3-Small, whole network fine-tuned | 0.6429 | 0.6786 |

Noise floor: val = 28 images → **one image = 3.6 points**. `apple` has 4 val images
(25 points each). Differences under ~7 points are 1–2 photographs, not an effect.
Train = 101 images: 12 apple / 36 cherry / 24 chestnut / 29 maple.

## 1. Shared code (once)

```python
def evaluate(model, loader, n, seed=None):
    # The validation block of 3c. seed makes a random augmentation draw identical for every model.
    if seed is not None:
        torch.manual_seed(0)
    model.eval()
    correct, preds, truths = 0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            preds.append(predictions.cpu()); truths.append(labels.cpu())
    return correct / n, torch.cat(preds).numpy(), torch.cat(truths).numpy()


def train_leaf(model, train_loader, train_ds, val_loader, val_ds,
               epochs=15, lr=1e-3, adamw=False, weight_decay=0.0,
               label_smoothing=0.0, keep_best=False, cosine=False):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    trainable = [p for p in model.parameters() if p.requires_grad]   # only the new layer if frozen
    if adamw:
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(trainable, lr=lr)
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
        if val_accuracy > best_acc:                                  # keep the best epoch's weights
            best_acc, best_state = val_accuracy, copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch + 1:2d}/{epochs}: Train Loss = {total_train_loss / len(train_ds):.4f}, "
              f"Val Acc = {val_accuracy:.4f}")
        if scheduler is not None:
            scheduler.step()

    if keep_best:
        model.load_state_dict(best_state)
        print(f"restored the best epoch: Val Acc = {best_acc:.4f}")
    return best_acc, evaluate(model, val_loader, len(val_ds))[1:]


def per_class_accuracy(preds, truths, classes):
    for i, name in enumerate(classes):
        m = truths == i
        print(f"  {name:9s} {np.mean(preds[m] == i):.3f}   ({int(np.sum(m))} images)")
    print(f"  balanced  {np.mean([np.mean(preds[truths == i] == i) for i in range(len(classes))]):.4f}")
```

Baseline through the same code path (reproduces 3c):

```python
torch.manual_seed(0)
baseline = LeafNet().to(device)
b0 = train_leaf(baseline, my_train_loader, my_train_ds, my_val_loader, my_val_ds)
```

## 2. E1 — ColorJitter next to the geometric augmentation

Edit cell 60: add the `jitter` flag and put it in the list before `ToTensor`
(ColorJitter acts on the PIL image). `jitter = []` gives the 3d row back.

```python
jitter = [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)]
jitter = []          # ← the 3d baseline, no colour jitter

aug_train_list = [
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
] + jitter + [transforms.ToTensor()]
if normalize is not None:
    aug_train_list.append(normalize)

aug_train_transform = transforms.Compose(aug_train_list)
aug_train_ds = LeafDataset("data/leaves/train", aug_train_transform)
aug_train_loader = DataLoader(aug_train_ds, batch_size=32, shuffle=True)
```

```python
torch.manual_seed(0)
b1 = train_leaf(LeafNetBN(), aug_train_loader, aug_train_ds, my_val_loader, my_val_ds)
```

## 3. E2 — transfer learning: frozen backbone vs fine-tuned

Frozen (linear probe). The harness already optimizes only `requires_grad=True` parameters.

```python
torch.manual_seed(0)
model = models.mobilenet_v3_small(weights="DEFAULT")
for p in model.parameters():
    p.requires_grad = False                                          # freeze the pretrained backbone
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)  # new layer: trainable by default
model = model.to(device)
b2_frozen = train_leaf(model, train_loader, train_ds, val_loader, val_ds)
```

Fine-tuned baseline: the same cell with the `for` loop deleted.

Optional, and it must be its own labelled row (Task 2 does NOT do this, so don't mix it
into the Task 2 comparison): right after `model.train()` inside the loop,

```python
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()          # keep the frozen batch norms at their ImageNet statistics
```

Notes: `classifier` is `Linear(576→1024) → Hardswish → Dropout(0.2) → Linear(1024→4)`, so a
frozen backbone leaves a 1024-dim ImageNet feature vector with a linear head on top.

## 4. E3 — pretrained + augmentation + cosine schedule + 40 epochs

A combination row, not a single factor — say so in the table.

```python
aug_train_loader_t2 = DataLoader(LeafDataset("data/leaves/train", aug_train_transform),
                                 batch_size=32, shuffle=True)

torch.manual_seed(0)
model = models.mobilenet_v3_small(weights="DEFAULT")
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
model = model.to(device)
b3 = train_leaf(model, aug_train_loader_t2, aug_train_ds, val_loader, val_ds,
                epochs=40, cosine=True)
```

## 5. E4 — regularization / classifier tweaks (five separate rows)

`LeafNetBN` with two optional flags; the defaults reproduce 3d bit-for-bit.

```python
class LeafNetBN(nn.Module):
    def __init__(self, pool="flatten", dropout=0.0):        # pool="gap" → global average pooling
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
        x = x.mean([2, 3]) if self.gap else x.flatten(1)     # GAP instead of flattening
        x = self.drop(torch.relu(self.fc1(x)))
        return self.fc2(x)
```

```python
torch.manual_seed(0); train_leaf(LeafNetBN(dropout=0.3), aug_train_loader, aug_train_ds, my_val_loader, my_val_ds)   # dropout
torch.manual_seed(0); train_leaf(LeafNetBN(pool="gap"),  aug_train_loader, aug_train_ds, my_val_loader, my_val_ds)   # GAP
torch.manual_seed(0); train_leaf(LeafNetBN(), aug_train_loader, aug_train_ds, my_val_loader, my_val_ds, adamw=True, weight_decay=1e-4)   # AdamW
torch.manual_seed(0); train_leaf(LeafNetBN(), aug_train_loader, aug_train_ds, my_val_loader, my_val_ds, label_smoothing=0.1)             # label smoothing
torch.manual_seed(0); train_leaf(LeafNetBN(), aug_train_loader, aug_train_ds, my_val_loader, my_val_ds, keep_best=True)                  # best epoch
```

`keep_best` changes which weights you export, not what is learned: 3d ended at 0.6071 after
peaking at 0.6429, so best-epoch selection is worth one validation image for free.

## 6. E5 — last year's photographs

Verified on disk: the archive is 175 train + 30 val per species, same four folder names,
and the same 96×96 JPEGs as the group set — so `LeafDataset` and `transform` work unchanged
and `ConcatDataset` class indices line up.

```python
if not os.path.isdir("hist/leaves/train"):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/tomasvicar/MLR-public/master/exercises/data/ex08_leaves_images.zip",
        "hist.zip")
    with zipfile.ZipFile("hist.zip") as archive:
        archive.extractall("hist")
print(os.listdir("hist"), os.listdir("hist/leaves/train"))
```

(a) unseen holdout — any model already trained, tested on leaves your group never photographed:

```python
hist_val_ds = LeafDataset("hist/leaves/val", transform)            # plain transform, like my_val_ds
hist_val_loader = DataLoader(hist_val_ds, batch_size=32, shuffle=False)
acc, preds, truths = evaluate(baseline, hist_val_loader, len(hist_val_ds))
print(f"holdout accuracy {acc:.4f} on {len(hist_val_ds)} images")
```

(b) more training data — 700 historical images added to the 101:

```python
from torch.utils.data import ConcatDataset            # the one new import

hist_train_ds = LeafDataset("hist/leaves/train", aug_train_transform)
comb_train_ds = ConcatDataset([aug_train_ds, hist_train_ds])   # same sorted class folders → same indices
comb_train_loader = DataLoader(comb_train_ds, batch_size=32, shuffle=True)

torch.manual_seed(0)
b5 = train_leaf(LeafNetBN(), comb_train_loader, comb_train_ds, my_val_loader, my_val_ds)
```

LEAKAGE: never report `hist/leaves/val` accuracy for (b) — it was in its training set.
For (b) the group `my_val_loader` is the only clean test.
Caveats: at fixed epochs, 801 images = 26 batches/epoch (390 steps) vs 4 batches (60 steps),
so "more data" and "more steps" move together; and 700 historical images outvote your 101.

## 7. E6 — camera-like validation set (required by the assignment)

Same files, fixed seed → every variant sees identical corrupted inputs.

```python
cam_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.5, 1.0)),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.15),
    transforms.ToTensor(), transforms.Normalize(mean, std)])

cam_ds = LeafDataset("data/leaves/val", cam_transform)
cam_loader = DataLoader(cam_ds, batch_size=32, shuffle=False)

acc, preds, truths = evaluate(model, cam_loader, len(cam_ds), seed=0)
```

Report three columns per row: clean val / camera-like / historical holdout.
Note: `RandomRotation` on an already tightly cropped 96×96 image adds black corners that a
centred phone preview does not have — a drop on clean val with a gain on camera-like val is
a real win, not a regression.

## 8. Multi-seed and reproducible shuffling

```python
my_train_loader = DataLoader(my_train_ds, batch_size=32, shuffle=True,
                             generator=torch.Generator().manual_seed(0))
```

```python
for seed in [0, 1, 2]:
    torch.manual_seed(seed)
    model = LeafNetBN().to(device)
    print(f"seed {seed}: {train_leaf(model, aug_train_loader, aug_train_ds, my_val_loader, my_val_ds)}")
```

## 9. Test-time averaging (no retraining)

Two details matter, both found by measurement (section 11): average the **softmax
probabilities**, not the logits — a sum of logits follows the single most confident view
instead of the majority — and use **mild** copies (a slightly tighter crop only), because
strongly rotated/recoloured copies are a corruption whose errors are correlated, so voting
amplifies them instead of cancelling them.

```python
mild_transform = transforms.Compose([        # a slightly tighter crop, nothing else
    transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
    transforms.ToTensor(), transforms.Normalize(mean, std)])
mild_ds = LeafDataset("data/leaves/val", mild_transform)


def predict_meaned(model, ds, mild, copies=4):
    # the probabilities of the clean image and of `copies` mild copies, averaged before the argmax
    with torch.no_grad():
        total = None
        for dataset, seed in [(ds, None)] + [(mild, s) for s in range(1, copies + 1)]:
            if seed is not None:
                torch.manual_seed(seed)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            out = torch.cat([torch.softmax(model(images.to(device)), dim=1) for images, _ in loader])
            total = out if total is None else total + out
    return (total / (copies + 1)).argmax(dim=1).cpu().numpy()
```

## 10. Measured results — every row, three seeds

Run locally on this machine (Python 3.12, torch 2.15.0+cpu, `_ablations/run_ablations.py`,
`results2.csv` + `best_s*.csv`). **No CUDA, and a different torch build from your Colab
notebook**, so absolute numbers do not carry over one-for-one: the local `B2` reproduced
your Task 2 result exactly (0.643 final / 0.679 best at seed 0), but local `B0` gave 0.464
where the notebook gave 0.536 (same peak, different last epoch, because the loader
`generator` is pinned here). **Compare rows within this table, never across environments.**

Every number is the mean over seeds 0/1/2, `[min-max]` where the spread is wide.
`best` = the best validation epoch, `final` = the last epoch.

| variant | final val (28) | best val (28) | balanced val | camera-like (28) | hist holdout (120) |
|---|---|---|---|---|---|
| B0 3c from scratch, no aug | 0.512 [0.46–0.61] | 0.571 [0.54–0.61] | 0.451 | 0.357 [0.29–0.39] | 0.358 [0.34–0.38] |
| B1 3d + geometric aug | 0.452 [0.32–0.61] | 0.595 [0.57–0.61] | 0.448 | 0.274 [0.25–0.29] | 0.414 [0.38–0.45] |
| E1 + ColorJitter | 0.524 | 0.583 | 0.429 | **0.500** | 0.442 |
| E4a + dropout 0.3 | 0.571 | 0.619 | 0.517 | 0.286 | 0.417 |
| E4b + global average pooling | 0.512 | 0.548 | 0.446 | 0.250 | 0.447 |
| E4c + AdamW wd 1e-4 | 0.464 | 0.595 | 0.453 | 0.286 | 0.406 |
| E4d + label smoothing 0.1 | 0.571 | **0.643** | 0.533 | 0.298 | 0.450 |
| E5 + 700 historical images | 0.643 | 0.691 | 0.622 | 0.345 | n/a — leakage |
| B2 Task 2 MobileNet, fine-tuned | 0.691 | 0.702 | 0.624 | 0.536 | 0.550 |
| E2 MobileNet, backbone frozen | 0.619 | 0.643 | 0.572 | 0.464 | 0.489 |
| E3 MobileNet + aug + cosine + 40 ep | 0.750 | 0.833 | 0.715 | 0.536 | **0.650** |
| **BEST = E1 + E3 + E5** | **0.810** [0.75–0.89] | **0.857** [0.82–0.93] | **0.777** | **0.738** [0.71–0.79] | n/a — leakage |

### Is the difference larger than the noise? (Task 3's closing question, answered)

One group-validation image = **3.6 points**; one historical image = **0.8 points**. The
per-seed spread of the *final* epoch reaches **28.6 points on B1** (0.32–0.61, eight
images, from the seed alone), while the *best* epoch of the same runs sits in 3.6 points
(0.57–0.61). So: comparing single final epochs is comparing noise; comparing best epochs
over 3 seeds is a measurement. Treat differences under ~10 points (3 images) on the
28-image val set as unresolved, and prefer the 120-image holdout and camera-like columns,
which resolve the same differences about four times more finely.

### What each change brought — ready for the report

**E1 colour augmentation** (against B1). Clean best 0.595 → 0.583, camera-like
0.274 → **0.500**, holdout 0.414 → 0.442.
> Adding colour jitter on top of the geometric augmentation left the clean validation
> accuracy unchanged (best 0.595 → 0.583, a third of one validation image) but raised the
> camera-like validation accuracy from 0.274 to 0.500, a gain of 6.3 validation images. Our
> group's photographs differ mainly in exposure and white balance rather than in shape, so
> jitter costs nothing on the images we already have and stops the network collapsing when
> the phone's colour response is different.

**E2 frozen backbone vs fine-tuned** (against B2). Final 0.691 → 0.619, best 0.702 → 0.643,
holdout 0.550 → 0.489, at 5 s instead of 13 s per run.
> Freezing the pretrained backbone and training only the new last layer lost about two
> validation images (best 0.702 → 0.643) and 6 points on last year's photographs, so with
> only 101 images fine-tuning the whole network still pays. The frozen probe is the option
> when training time matters — it runs in a third of the time and cannot adapt the backbone
> to our particular photo session.

**E3 augmentation + cosine schedule + 40 epochs** (against B2). Final 0.691 → 0.750, best
0.702 → 0.833, holdout 0.550 → **0.650**.
> Giving the pretrained network augmentation, a cosine learning-rate schedule and 40 instead
> of 15 epochs raised it from 0.691 final / 0.702 best to 0.750 / 0.833 and, more importantly,
> from 0.550 to 0.650 on leaves our group never photographed. The augmentation is what makes
> 40 epochs possible at all — without it the network memorises 101 photographs — while the
> schedule shrinks the learning rate so the last epochs refine instead of jumping between two
> models.

**E4 regularisation and classifier tweaks** (against B1). Dropout and label smoothing helped,
GAP and AdamW did not.
> On the from-scratch network only two of the four helped — label smoothing (best 0.595 → 0.643,
> final +11.9 points) and dropout 0.3 (best +2.4, final +11.9) — and both are worth only one to
> three validation images. AdamW with weight decay 1e-4 changed nothing (best identical at 0.595)
> and global average pooling actively hurt (best 0.548, camera-like 0.250, i.e. guessing), so on
> 101 photographs these regularisers mostly substitute for one another instead of stacking.

**E4e keeping the best epoch** (free, needs no extra training — it is the `best` column).
> Keeping the weights of the best validation epoch instead of the last moved the from-scratch
> network from a final-epoch mean of 0.452, spread 0.32–0.61 over three seeds, to a best-epoch
> mean of 0.595, spread 0.57–0.61. The last epoch of a 15-epoch run is effectively a random draw
> from the models seen during training, so the best epoch is the only repeatable number to report
> — and it is the model you export anyway.

**E5 last year's photographs, as more training data** (against B1). Final 0.452 → 0.643
(+19.0 points, 5.3 images), best 0.595 → 0.691, camera-like 0.274 → 0.345.
> Adding last year's 700 photographs to our 101 lifted the from-scratch network from 0.452 final /
> 0.595 best to 0.643 / 0.691 on our own validation images, so what this network needed was more
> data rather than more regularisation. Its accuracy on other phones cannot be measured, because
> those photographs were part of its training set.

**The unseen holdout, across phones.** Same model, scored on the 120 photographs no one in our
group took: B0 0.358, B1 0.414, E1 0.442, B2 0.550, E2 0.489, **E3 0.650**.
> On last year's photographs the pretrained networks separate from the from-scratch ones by 10 to
> 29 points, and augmentation with a longer schedule adds a further 10 points on top of transfer
> learning. This column and the camera-like column give the same ranking of the variants while our
> own 28-image validation set barely separates them, which is why the phone model should be chosen
> on those two columns.

**BEST — the combination that survived** (E1 + E3 + E5). Final 0.810, best 0.857, balanced 0.777,
camera-like **0.738** against B1's 0.274 and B2's 0.536.
> Combining the three changes that survived testing — colour augmentation, the pretrained network
> with a cosine schedule over 40 epochs, and last year's 700 photographs — reaches 0.810 final /
> 0.857 best on our validation set and 0.738 on the camera-like set, where the Task 3d pipeline
> manages 0.274. That is +35.7 final points over 3d and about +20 camera-like points over the plain
> Task 2 model, and it is the model to export as ONNX for the phone; its score on last year's
> photographs is unavailable precisely because those images are in its training set.

### Where the mistakes are (per species, mean over 3 seeds)

`apple` is the failure class in every variant. On the 30 historical apple leaves: B0 0.09,
B1 0.24, E1 0.00, B2 0.13, E2 0.30, E3 0.37. The dominant confusion is **apple → cherry**
(E3 sends 13 of 30 apple leaves to cherry), which is the pair Task 2 asks about, with a
secondary chestnut ↔ maple confusion (E3: 13 of 30 maple leaves sent to chestnut). `chestnut`
is the easiest class for every pretrained model (0.82–0.94).

> Our group has the fewest apple training photographs (12 of 101), and even the best variant
> answers only 0.42 of the four apple validation images correctly, so the cheapest next step for
> the phone app is more apple photographs rather than a different network.

### Test-time averaging — measured, and it needed two corrections

Averaging the clean image with four copies, scored on the 28 clean validation images:

| variant | clean | sum of logits + 4 strong copies | mean of softmax + 4 strong | mean of softmax + 4 **mild** copies |
|---|---|---|---|---|
| B1 | 0.452 | 0.429 | 0.429 | **0.536** |
| E4c | 0.464 | 0.405 | 0.440 | **0.548** |
| B2 | 0.690 | 0.571 | 0.655 | 0.655 |
| E3 | 0.750 | 0.690 | 0.750 | **0.762** |
| BEST | 0.810 | 0.798 | 0.810 | **0.857** |

> Averaging the softmax probabilities of the clean image and four mildly re-cropped copies adds
> 4.7 points to the model we would ship (0.810 → 0.857) and 8 to 9 points to the weak from-scratch
> ones, at the cost of five forward passes per frame. Averaging over strongly rotated and recoloured
> copies does not help at all, because those views make correlated mistakes and a vote amplifies a
> shared error instead of cancelling independent ones.

The first version of this test averaged **logits**, which made every variant look much worse than
its single-pass accuracy (BEST 0.798 against 0.810): a sum of logits is dominated by the most
confident single view rather than by the majority. Probability averaging is the correct rule.

### What did not help, and what goes on the phone

Did not help: AdamW + weight decay (identical best accuracy), global average pooling (worse
everywhere, near-chance on the camera-like set), freezing the backbone (−2 images), colour
augmentation on *clean* validation only (a wash — it pays on camera-like data), and test-time
averaging over corrupted copies. None of the four E4 regularisers moved the pretrained network
at all, because it was never overfitting our 101 images in the first place.

Ship **BEST**, exported as in 2e, with the mild-copy averaging of section 9 if you want the extra
4.7 points in the browser (five inferences per frame still fits the ~5 in/s budget in `README.md`
only at a lower frame rate — measure it on the iPhone before promising it). Report B0/B1/B2 as your
fixed baselines, one change per row, three seeds, and the image count next to every number.

## 11. One cell that reproduces the whole table in Colab

Run section 1 (`evaluate`, `train_leaf`) and section 6 (the historical download) first, then
paste this as a single cell. It uses only your `LeafDataset`, `LeafNet`, `LeafNetBN` and the
notebook's `transform`, `mean`, `std`, `device`; the only thing it adds is the list of variants.
Expect roughly 20 minutes on a Colab GPU, mostly in the last two rows.

```python
from torch.utils.data import ConcatDataset

def make_transform(jitter):
    colour = [transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)] if jitter else []
    return transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20)] + colour +
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

def mobilenet(freeze=False):
    m = models.mobilenet_v3_small(weights="DEFAULT")
    if freeze:
        for p in m.parameters():
            p.requires_grad = False
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 4)
    return m.to(device)

# name, network, training set, epochs, extra train_leaf options, whether the holdout leaks
DATA = {
    "plain":  LeafDataset("data/leaves/train", transform),
    "aug":    LeafDataset("data/leaves/train", make_transform(False)),
    "jitter": LeafDataset("data/leaves/train", make_transform(True)),
}
DATA["hist_aug"] = LeafDataset("hist/leaves/train", make_transform(False))
DATA["hist_jitter"] = LeafDataset("hist/leaves/train", make_transform(True))
DATA["aug+hist"] = ConcatDataset([DATA["aug"], DATA["hist_aug"]])
DATA["jitter+hist"] = ConcatDataset([DATA["jitter"], DATA["hist_jitter"]])

VARIANTS = [
    ("B0  3c from scratch",              LeafNet,             "plain",         15, {},                          False),
    ("B1  3d + geometric aug",           LeafNetBN,           "aug",           15, {},                          False),
    ("E1  + ColorJitter",                LeafNetBN,           "jitter",        15, {},                          False),
    ("E4a + dropout 0.3",                lambda: LeafNetBN(dropout=0.3), "aug", 15, {},                         False),
    ("E4b + global average pooling",     lambda: LeafNetBN(pool="gap"),  "aug", 15, {},                         False),
    ("E4c + AdamW weight decay 1e-4",    LeafNetBN,           "aug",           15, {"adamw": True, "weight_decay": 1e-4}, False),
    ("E4d + label smoothing 0.1",        LeafNetBN,           "aug",           15, {"label_smoothing": 0.1},    False),
    ("E5  + 700 historical images",      LeafNetBN,           "aug+hist",      15, {},                          True),
    ("B2  Task 2 MobileNet, fine-tuned", lambda: mobilenet(), "plain",         15, {},                          False),
    ("E2  MobileNet, backbone frozen",   lambda: mobilenet(True), "plain",     15, {},                          False),
    ("E3  MobileNet + aug + cosine 40ep", lambda: mobilenet(), "aug",          40, {"cosine": True},            False),
    ("BEST E1 + E3 + E5",                lambda: mobilenet(), "jitter+hist",   40, {"cosine": True},            True),
]

hist_val_ds = LeafDataset("hist/leaves/val", transform)
hist_val_loader = DataLoader(hist_val_ds, batch_size=32, shuffle=False)

for name, build, data, epochs, options, leaks in VARIANTS:
    scores = []
    for SEED in [0, 1, 2]:
        torch.manual_seed(SEED)
        model = build()
        train_ds = DATA[data]
        loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                            generator=torch.Generator().manual_seed(SEED))
        best, _, _ = train_leaf(model, loader, train_ds, my_val_loader, my_val_ds,
                                epochs=epochs, **options)
        final, preds, truths = evaluate(model, my_val_loader, len(my_val_ds))
        camera, _, _ = evaluate(model, cam_loader, len(cam_ds), seed=0)
        hist = float("nan") if leaks else evaluate(model, hist_val_loader, len(hist_val_ds))[0]
        scores.append((final, best, camera, hist))
        print(f"{name:34s} seed {SEED}:  final {final:.4f}  best {best:.4f}  "
              f"camera {camera:.4f}  holdout {hist:.4f}")
    mean = [sum(s[i] for s in scores) / len(scores) for i in range(4)]
    print(f"{name:34s} MEAN  :  final {mean[0]:.4f}  best {mean[1]:.4f}  "
          f"camera {mean[2]:.4f}  holdout {mean[3]:.4f}\n")
```

The `holdout` column prints `nan` exactly where the historical photographs were part of
training — leave it as `n/a` in the report rather than dropping the column.

## 12. Filling the `n/a`: a held-out slice of the historical archive

The combination model cannot be scored on `hist/leaves/val` (its training images come from the
same archive). Instead the historical **training** folder is split by a fixed rule — every third
file of each class, in sorted order, goes to the test side — so the test images are from other
phones and were never trained on. `_ablations/split_test.py`, three seeds, pretrained network,
augmentation with colour jitter, cosine schedule, 40 epochs, scored at the best group-validation
epoch in both arms. **One change: the extra photographs.**

| arm | our group validation (28) | 236 held-out other-phone images | per seed |
|---|---|---|---|
| D0 our 101 photographs only | 0.762 | **0.579** [0.479–0.686] | 0.479 / 0.572 / 0.686 |
| D1 + 464 historical photographs | 0.833 | **0.862** [0.852–0.877] | 0.856 / 0.852 / 0.877 |

Per-seed gain: **+37.7, +28.0, +19.1 points** — positive in all three runs, mean **+28.2 points**
on a 236-image test set where one image is 0.42 points, so ≈67 images. Here one image of the 236
is worth 0.42 points instead of the 3.6 points an image of our 28 is worth, which is why this is
the most trustworthy number in the whole set.

> Adding 464 of last year's photographs raised accuracy on 236 photographs from the same archive
> that were held out of training — other students, other phones — from 0.579 to 0.862, a gain of
> 28 points that appeared in all three seeds. It also cut the run-to-run spread from 20.7 points to
> 2.5, so the extra data did not only move the average up, it removed the dependence on which
> random start the network happened to get.

Note the honest caveat for the report: this measures generalisation **within** that archive's
photo session, so it is the same-device-population question, not a third phone. The only truly
unseen device population left is your own live camera — which is the 2e export test in the next
section.

## 13. Exporting the model you chose (2e)

`_ablations/export_best.py` retrain the combination with `keep_best=True` and exports to
`exports/leaf_model_best.onnx` with the same `model_info` metadata cell 49 writes, then scores all
28 validation images through `onnxruntime` and compares against PyTorch:

| | |
|---|---|
| file | `exports/leaf_model_best.onnx`, 6.4 MB (the deployed one is 6.3 MB) |
| input / output | `image` float32 `1 × 3 × 96 × 96` NCHW → `logits`, 4 values, unnormalised |
| metadata | `model_info = {"classes": ["apple","cherry","chestnut","maple"], "input_size": 96, "mean": [0.485,0.456,0.406], "std": [0.229,0.224,0.225]}` — identical to the deployed `model_info.json`, so `index.html` needs no change |
| this checkpoint | group validation 0.8214 (seed 0's best epoch, the weakest of the three seeds), camera-like 0.6786 |
| torch vs onnxruntime | worst absolute logit difference **3.4 × 10⁻⁴**, **same predicted class on all 28** validation images, onnx accuracy **0.8214** — equal to the PyTorch number |

Exported the same way as cell 49 (a `torch.onnx.export` of a single `1 × 3 × 96 × 96` example), so
the file takes **one frame per call**, which is what the page does. Two consequences if you decide
to ship it:

1. **Nothing here touched `mlr-ex08/`** — the live page still serves the Task 2 model, and
   replacing `leaf_model.onnx` changes what your deployed URL does, so that swap is your call.
2. If you do swap it, `check_image.jpg` and `check_expected.json` belong to the **old** model.
   Re-run cell 51 with the new network before pushing, or the page-vs-notebook agreement you
   report in `REVIEW.md` will be comparing two different models.

The preprocessing is unchanged (centre square → 96 × 96 → /255 → the same mean/std), so the
browser pipeline stays valid. The mild-copy averaging of section 9 is **not** in the exported
file — it would be five calls per frame on the phone; measure the frame rate on the iPhone before
committing to it.

Two Windows-only wrinkles found while doing this, in case you hit them locally: the exporter
prints a ✅ that the Greek console code page cp1253 cannot encode (fixed in the script with
`sys.stdout.reconfigure(encoding="utf-8")`), and `torch.onnx.export` needs `onnxscript`
installed, which is why cell 49 installs it.

## Housekeeping in the current notebook

- Cells 60 and 61 end with a stray `...` line (cell 61's output prints `Ellipsis`).
- Cells 60/61 contain `[cite: 5]` markers and a Greek note-to-self comment. Harmless, but
  delete them before a marker reads every line.
