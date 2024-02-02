# License Plates Recognition
## Technion - Israel Institute of Technology<br/>Department: The school of continuing education<br/>Course: Technion Certified Data Scientist

**Course final project:** Developing of neural network that recognizes car license plates.<br/>
**Dataset:** Cars images containing license plates.<br/>
**Programming Language:** Python 3.10.<br/>
**Technologies/Methods:** Faster R-CNN, PyTorch.<br/>
**Software:** Spyder IDE 5.4.1, Jupiter Notebook, Kaggle,<br/>
&emsp;&emsp;via-2.0.12 - VGG Image Annotator - an image annotation tool that can be used to<br/>
&emsp;&emsp;define regions in an image and create textual descriptions of those regions.

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
