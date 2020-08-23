"""
This script is used to make ensemble predictions from
various test predictions as final prediction
"""

import os
import numpy as np
import pandas as pd


## ensemble : sub_ens_15.csv
## Best Local Validation score
paths = [
         './tesPred_tta_fold0_[Xception-299]_aug2.csv',
         './tesPred_tta_fold0_[Xception-299]_aug2_ocr.csv',
         './tesPred_tta_fold0_[EfficientNetB3-299]_aug.csv',
         './tesPred_tta_fold0_[EfficientNetB3-299]_aug2_ocr.csv',
         './tesPred_tta_fold0_[EfficientNetB5-299]_aug.csv',
         './tesPred_tta_fold0_[InceptionResNetV2-299]_aug.csv',
         './tesPred_tta_fold0_[InceptionResNetV2-299-[focal_g2_a.25]]_aug.csv',
         './tesPred_tta_fold0_[InceptionResNetV2-299]_aug2.csv',
         './tesPred_tta_fold0_[InceptionResNetV2-299]_aug2_ocr.csv',
        ]

for idx, i in enumerate(paths):
    tes_pred = pd.read_csv(i)
    if(idx==0):
        pred_sub = tes_pred.drop(['filename'], axis=1).values / len(paths)
        filenames = tes_pred['filename']
    else:
        pred_sub += tes_pred.drop(['filename'], axis=1).values / len(paths)
    del tes_pred
pred_sub = np.argmax(pred_sub, axis=1)
sub_df = pd.DataFrame()
sub_df['filename'] = filenames
sub_df['category'] = pred_sub
sub_df['category'] = sub_df['category'].apply(lambda x: '{:02d}'.format(x))
sub_df.to_csv('sub_ens_15.csv', index=False)

## ensemble : sub_ens_13.csv
## Best Public LB score
paths = [
         './tesPred_tta_fold0_[Xception-299]_aug.csv',
         './tesPred_tta_fold0_[Xception-299]_aug2.csv',
         './tesPred_tta_fold0_[EfficientNetB3-299]_aug.csv',
         './tesPred_tta_fold0_[EfficientNetB5-299]_aug.csv',
         './tesPred_tta_fold0_[InceptionResNetV2-299]_aug.csv',
         './tesPred_tta_fold0_[InceptionResNetV2-299-[focal_g2_a.25]]_aug.csv',
         './tesPred_tta_fold0_[InceptionResNetV2-299]_aug2.csv',
         './tesPred_tta_fold0_[InceptionResNetV2-299]_aug2_ocr.csv',
        ]

for idx, i in enumerate(paths):
    tes_pred = pd.read_csv(i)
    if(idx==0):
        pred_sub = tes_pred.drop(['filename'], axis=1).values / len(paths)
        filenames = tes_pred['filename']
    else:
        pred_sub += tes_pred.drop(['filename'], axis=1).values / len(paths)
    del tes_pred
pred_sub = np.argmax(pred_sub, axis=1)
sub_df = pd.DataFrame()
sub_df['filename'] = filenames
sub_df['category'] = pred_sub
sub_df['category'] = sub_df['category'].apply(lambda x: '{:02d}'.format(x))
sub_df.to_csv('sub_ens_13.csv', index=False)
