1). It runs on your phone:

First, I opened the live GitHub Pages link on my phone and allowed access to the camera. Everything worked straight away. The camera opened without any issues, and the live video appeared clearly at full resolution with no noticeable lag or stuttering.
I also tested it using leaf images displayed on my computer screen. The model immediately recognized them correctly, predicting maple with 92.8% confidence and cherry with 98.5%. The progress bars also updated smoothly in real time, showing the results as they changed.

!Screenshot_20261105_193918_Samsung_Browser.jpg








2).The same numbers as the notebook:

To make sure the web application was producing the same results as the original model, I uploaded `check_image.jpg` and compared its predictions with the values stored in `check_expected.json`.
The results were almost exactly the same as those from the notebook:

- Apple: 0.8296
- Cherry: 0.1679
- Chestnut: 0.0025
- Maple: 0.0000037

The largest difference between the web application and the notebook was only around \(2.8 \times 10^{-6}\). This is extremely small and comfortably meets the requirement of matching the predictions to three decimal places.

3).The preprocessing matches your model: 

I also checked how images are prepared before being passed to the model by looking through the `index.html` file.
The `sourceToTensor` function first takes a square crop from the center of the image and resizes it to 96 × 96 pixels using high-quality canvas smoothing.
After that, the pixel values are divided by 255 and normalized using the exact mean and standard deviation values from `model_info.json`:

- Mean: [0.485, 0.456, 0.406]
- Standard deviation: [0.229, 0.224, 0.225]

Finally, the image data is converted into a planar `[1, 3, 96, 96]` float32 tensor.
This means that the web application follows the same preprocessing steps used in the original PyTorch notebook, ensuring that the model receives images in the format it was trained to work with.

4).Nothing leaves the phone:

One important thing I checked was whether the application sends any images or camera footage to an external server.
The good news is that all predictions are made directly on the device using WebAssembly. This means that neither camera frames nor uploaded images are sent anywhere.
I also checked the network activity to see what the application actually loads. It only downloads the necessary files when the page starts, including the ONNX model, `model_info.json`, and the ONNX and QR-code libraries from jsDelivr.
Even the QR code displayed at the bottom of the page is generated locally using an HTML5 canvas, rather than relying on an external QR-code service.
Overall, the application performs its image recognition directly on the device, without needing to upload personal images or camera footage to a server.
