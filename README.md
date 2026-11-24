# License Plates Recognition
*Course final project · Technion Certified Data Scientist · [Technion – Israel Institute of Technology](https://www.technion.ac.il/) · 2024*

Developing of neural network that recognizes car license plates.<br/>
**Dataset:** Cars images containing license plates.<br/>
**Programming Language:** Python 3.10.<br/>
**Technologies/Methods:** Faster R-CNN, PyTorch.<br/>
**Libraries:** PyTorch, torchvision, torch_snippets, OpenCV, NumPy, pandas, Matplotlib, scikit-learn, Pillow.<br/>
**Installation:** `pip install -r requirements.txt` (full list in [requirements.txt](requirements.txt)).<br/>
**Software:** Spyder IDE 5.4.1, Jupiter Notebook, Kaggle,<br/>
&emsp;&emsp;via-2.0.12 - VGG Image Annotator - an image annotation tool that can be used to<br/>
&emsp;&emsp;define regions in an image and create textual descriptions of those regions.

## Results

The model detects each digit of the plate separately (10 classes, 0–9); the digits are then
sorted from left to right to build the car number. Two metrics are therefore reported:

- **Digit recognition** – share of digits recognized correctly, compared with the real number.
- **Plate recognition** – share of plates where the whole number is correct. A plate counts as
  wrong if even one of its 7–8 digits is wrong or missing, so this metric is always lower.

Evaluated on 50 test plates (`images_test.zip`, file name = real car number):

| Post-processing | Digit recognition | Plate recognition (exact) | Average similarity |
|---|---:|---:|---:|
| Original (NMS IoU 0.05) | 94.9% | 66% (33/50) | 0.967 |
| Improved (NMS IoU 0.3, score ≥ 0.5) | **98.1%** | **86% (43/50)** | 0.983 |

**Error analysis.** Most errors of the original version were a single missing digit,
especially when two equal digits stand next to each other (e.g. `11699902` → `1699902`).
Digit boxes on a plate stand close together and often overlap slightly; non-maximum
suppression with IoU threshold 0.05 treated such neighbours as duplicates and removed one of them.
Raising the threshold to 0.3 fixed this without retraining the model. Any threshold between
0.1 and 0.5 gives 42–44 correct plates, so the result is not tuned to the specific test images.
The remaining errors are real recognition mistakes (6↔8, 6↔5, a missed edge digit) and would
need more training data.

**Project files:**<br/>
* *License-plate-faster-rcnn-py.ipynb* - Model creation/training. The code is run on the Kaggle platform.<br/>
* *License_Plate_Faster_RCNN_prediction.py* - Testing/prediction.<br/>
* *Plot_Results.py* - Plot the results from the data frame is contained in the file.<br/>
* *CSV_convertation.py* - Convert the data received from the VGG Image Annotator to more appropriate format is used for training of the model..<br/>
* *Utils.py* - Some useful functions for image processing.<br/>
* *Images_padding_resizing.py* - Resizing images by padding.<br/>
* *df.csv* - Annotation data.<br/>
* *epochs_aver_df.csv* - Training/testing data for each epoch.<br/>
* *images.zip* - images for training.<br/>
* *images_test.zip* - images for testing.<br/>
* *Tests_example.pdf* - contains tests results/output.<br/>
* *Final Project - deep learning.pdf* - project report.<br/>

The dataset presented in the repository has only 10 examples (images.zip). The full dataset has 500 images.<br/>

See *"Final Project - deep learning.pdf"* for the details.
