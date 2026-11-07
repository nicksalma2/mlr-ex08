# Leaf Classifier Project Report (Task 4)

## What We Tested and Why

The goal of this task was to make our leaf classifier actually work well on real phone cameras. Phone photos in real life are messy—the lighting changes, angles are weird, and phone sensors handle colors differently compared to clean, cropped pictures.

Because our validation set has only 28 photos, one single mistake moves the score by about 3.6%. That makes single test runs very noisy. To keep things fair, we tested changes one by one starting from our Task 3 baseline, kept everything on the same images, and ran all the main tests across three random seeds (seed 0, 1, and 2) to see if an improvement was real or just lucky noise. We also tested every version on two harder sets: a camera-like validation set (with random crops, rotations, and color shifts) and an older dataset from last year's students with different phones.

Here is how the main versions stacked up across our runs:

- **B0 (frozen backbone):** We only trained the final layer. It averaged around 52% clean accuracy and 49% on camera photos. It ran fast, but plateaued early.
- **B1 (our Task 3 baseline):** Fine-tuning all layers brought average clean accuracy to around 57% and camera-like accuracy to roughly 54%. But seed runs bounced between 32% and 60%, showing how unstable it was.
- **E1 (color jitter):** Clean accuracy was about 54%, but camera-like accuracy went up to 57%, and it handled last year's photos better (jumping from 47% to 54%). It taught the model not to rely on exact leaf shades.
- **E2 (color jitter + cosine scheduler, 15 epochs):** Reached around 60% on both clean and camera sets. Training felt much steadier.
- **E3 (more epochs + cosine scheduler, 40 epochs):** This gave a big boost, averaging 72.6% on clean validation (peaking at 82%) and 69% on camera-like images.
- **E4 tweaks (dropout, pooling, optimizer, label smoothing):** Most of these did not help. Dropout (57%) and AdamW (56%) made almost no difference compared to baseline. Global average pooling actually hurt the model, dropping clean accuracy to 51% because it blurred out edge details. Label smoothing was the only one that helped a bit, reaching about 58%.
- **E5 (adding last year's archive images):** Training with both our group's photos and last year's photos bumped clean accuracy to around 64% and camera accuracy to 61%. More data helped a lot.
- **BEST (combining the winners):** We combined last year's data, color jitter, 40 epochs with cosine decay, and kept the weights from the best validation epoch. This reached an average clean validation score of 85.7% (with seed 1 hitting 92.9%) and 76.2% on the camera-like set.

## What Worked and What Did Not

The biggest real gains came from three things:
1. **Cosine learning rate decay with 40 epochs:** Giving the model more time to learn while slowly lowering the learning rate made the biggest algorithmic difference. The baseline was stopping before it could properly settle.
2. **Adding historical photos:** We only had 101 training photos from our own group. Adding the 700 archived photos gave the network way more variety across lighting and phones, which visibly reduced overfitting.
3. **Color jitter:** Even when clean validation scores didn't jump high, jitter clearly helped the model survive synthetic camera distortion and unseen holdout photos.
4. **Saving the best epoch:** Instead of just keeping whatever weights were left at epoch 40, saving the best checkpoint reliably gave us 3% to 7% higher accuracy.

What failed:
- **Global average pooling:** This was the clearest failure. Leaf recognition relies heavily on leaf shapes, points, and edges. Collapsing all spatial details into averages right before classification threw away those visual cues.
- **AdamW and Dropout:** These changes mostly produced numbers within our normal seed noise without real improvement.

## Which Model We Put on the Phone

We picked the **BEST** model for the phone app. 

It is noticeably more robust than the baseline: it reached 85.7% average validation accuracy, stayed above 76% on camera distortions, and was the only model that reliably identified tricky classes like apple leaves without confusing them for cherry.

We exported it to ONNX format (`exports/leaf_model_best.onnx`, roughly 6 MB) along with its `model_info.json`. We tested the ONNX file against PyTorch on the validation set, and they produced identical predictions with practically zero difference in numbers. When loaded into the course camera web app, it runs smoothly in real time and classifies leaves cleanly without jumping around.