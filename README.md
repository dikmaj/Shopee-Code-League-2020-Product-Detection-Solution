# Shopee Code League 2020: Product Detection [Open Category] - Solution

This repository contains solution **(Python Scripts)** from team **PoorGuySearchingForASugar** ***(22th Place [top 4%] Private Leaderboard)*** 
for the **Shopee Code League 2020: Product Detection [Open Category]** competition 
held on the Kaggle Inclass platform.

**Kaggle Competitions Link :** https://www.kaggle.com/c/shopee-product-detection-open

**Requirements:** \
Latest version of:
- [`Python`](https://www.python.org/)
- [`numpy`](https://numpy.org/) and [`pandas`](https://pandas.pydata.org/)
- [`scikit-learn`](https://scikit-learn.org/stable/)
- [`cv2`](https://pypi.org/project/opencv-python/)
- [`tensorflow`](https://www.tensorflow.org/), make sure to install with **GPU dependencies**.

Earlier version of:
- [`imgaug`](https://imgaug.readthedocs.io/en/latest/) < `0.4.0`, this is because in version `0.4.0` parameter name `random_state` was changed to `seed` 
while Kaggle (at that time) still use earlier version. You can adjust this parameter if you wish to use latest version.

Code for training models is marked with suffix name `models_xx`. 
These codes contain **different training schemes** and a code can be **reused to train different architecture (more inside code comments)**. \
Main difference on these codes:
- `models_type1.py`: Base code for training which use simple augmentation. Cross-entropy loss
- `models_type2.py`: Base code (same as `models_type1.py`) but with Focal Loss
- `models_type3.py`: Base code + Mixup or Cutmix Augmentation. Cross-entropy loss
- `models_type4.py`: Multi input (image and text from OCR) architecture + Mixup or Cutmix image Augmentation. Cross-entropy loss.

**Warning:** Results could be different **(undeterminstic)** since all code was ran in GPU with no seed initialization. \

`ocr_gen.py` can be used to generate OCR text data. These data also available publicly at [here](https://www.kaggle.com/ekojsalim/scl-product-detection-useful-data).

Finally, we ensemble various predictions from these models with `ensemble.py`.
