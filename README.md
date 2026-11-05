# Leaf Classifier — MLR Exercise 8

**Live page: <https://nicksalma2.github.io/mlr-ex08/>**

Classifies a leaf as **apple**, **cherry**, **chestnut** or **maple** straight from your
phone's rear camera, entirely in the browser. No server, no upload — the frames never
leave the device.

Open the link on an iPhone and tap **Start camera** (iOS asks for permission first).
If the camera is unavailable, use **…or choose a photo** instead — the same pipeline runs
on the still image.

There is also a QR code at the bottom of the page that encodes the URL above, so you can
scan it with the iPhone camera app to get there.

## What it runs

| | |
|---|---|
| Model | `leaf_model.onnx` (MobileNet-style CNN, 6.3 MB) |
| Runtime | [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) 1.21.0, WASM backend, from the jsDelivr CDN |
| Input | `image` — `float32`, shape `1 × 3 × 96 × 96`, NCHW |
| Output | `logits` — 4 values, softmaxed in the page to get probabilities |
| Classes | `["apple", "cherry", "chestnut", "maple"]` |
| Pages | GitHub Pages, served from the `main` branch |

## Preprocessing

Taken verbatim from [`model_info.json`](model_info.json), which the page reads at runtime:

```json
{"classes": ["apple", "cherry", "chestnut", "maple"], "input_size": 96,
 "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
```

1. Take the **central square** of the current frame (or photo).
2. **Resize** it to `96 × 96`.
3. Scale RGB pixels to **0–1** (`/255`).
4. **Standardise** each channel: `(x − mean[c]) / std[c]`.
5. Reshape to NCHW and feed `1 × 3 × 96 × 96`.

The model's output tensor is named `logits` and is not normalised, so the page applies
softmax before display. If a future export already emits probabilities, the page detects
that and skips the softmax.

## Accuracy against the reference

`check_image.jpg` classified **through the deployed web page** (canvas resize + WASM
inference in the browser), versus `check_expected.json`:

| class | expected | page | Δ |
|---|---|---|---|
| apple | 0.82958835 | 0.82959108 | 2.7e−06 |
| cherry | 0.16795219 | 0.16794943 | 2.8e−06 |
| chestnut | 0.00245582 | 0.00245579 | 2.7e−08 |
| maple | 0.00000370 | 0.00000370 | 7.9e−11 |

Worst difference **2.8 × 10⁻⁶** — i.e. agreement up to float32 rounding.

## Files

- `index.html` — the whole app: camera, preprocessing, inference, results, QR code.
- `leaf_model.onnx` — the trained model.
- `model_info.json` — the spec the page is driven by.

The QR code is drawn on a `<canvas>` in your browser by
[qrcode-generator](https://www.npmjs.com/package/qrcode-generator); no third-party
QR-image service is used, so the URL is never sent anywhere.

## Notes

- Camera access requires a secure context, which GitHub Pages provides over HTTPS.
  It also works from `http://localhost` for local development.
- onnxruntime-web runs single-threaded here because GitHub Pages does not send the
  COOP/COEP headers that `SharedArrayBuffer` needs; the model is small enough that this
  still runs at ~5 inferences/second.
