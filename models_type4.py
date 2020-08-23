"""
This script is used to trained Xception, EfficientNetB3, and InceptionResNetV2 on 299x299 image,
plus a simple Embedding feature from OCR texts on image data.

Data input for OCR can be used from public dataset here: https://www.kaggle.com/ekojsalim/scl-product-detection-useful-data
The script for generating these OCR data is also provided (ocr_gen.py)

Simple augmentation + [Mixup or Cutmix] on image (see below) is used for training and 
Test Time Augmentation (TTA) scheme on image is used for validation and test data

This script use GPU Accelerator
"""

## Uncomment this line if training EfficientNet to install dependencies
# !pip install -q efficientnet

import numpy as np
import pandas as pd
import os
import imgaug as ia
from imgaug import augmenters as iaa
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import cv2
import pickle
import string

## Uncomment lines according to model type
## If training Xception, use first line. If training InceptionResNetV2, use second line, etc.
from tensorflow.keras.applications.xception import Xception, preprocess_input
# from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
# from efficientnet.tfkeras import EfficientNetB3, preprocess_input

# Credits: https://www.kaggle.com/mpalermo/keras-pipeline-custom-generator-imgaug
class DataGenerator(tf.keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, images_paths, ocr_annots, labels, n_class, batch_size=64, img_dim=(128,128,3), 
                 shuffle=False, augment=False, mixup_alpha=0.4, cutmix_alpha=0.4):
        self.labels       = labels       # array of labels
        self.n_class      = n_class      # number of class
        self.images_paths = images_paths # array of image paths
        self.ocr_annots   = ocr_annots
        self.dim          = img_dim      # image dimensions
        self.batch_size   = batch_size   # batch size
        self.shuffle      = shuffle      # shuffle bool
        self.augment      = augment      # augment data bool
        self.mixup_alpha  = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # Generate one batch of data
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        
        # select ocr annot
        annots = self.ocr_annots[indexes]

        # select data and load images
        labels = self.labels[indexes]
        labels = tf.keras.utils.to_categorical(labels, num_classes=self.n_class)
        images = [cv2.imread(self.images_paths[k]) for k in indexes]
        images = [cv2.resize(img, self.dim[:2], cv2.INTER_AREA) for img in images]
        
        # preprocess and augment data
        if self.augment == True:
            images = self.augmentor(images)
            
            # dice to decide whether to apply mixup, cutmix, or pass
            dice = np.random.randint(0, 3)
            if(dice==0 or dice==1):
                # Choose random data for mixup or cutmix
                indexes_rn = self.indexes[~np.isin(self.indexes, indexes)]
                indexes_2 = np.random.choice(indexes_rn, len(indexes))

                # select data and load images
                labels_2 = self.labels[indexes_2]
                labels_2 = tf.keras.utils.to_categorical(labels_2, num_classes=self.n_class)
                images_2 = [cv2.imread(self.images_paths[k]) for k in indexes_2]
                images_2 = [cv2.resize(img, self.dim[:2], cv2.INTER_AREA) for img in images_2]
                if(dice==0):
                    images, labels = self.mixup(np.array(images), labels, np.array(images_2), labels_2)
                else:
                    images, labels = self.cutmix(np.array(images), labels, np.array(images_2), labels_2)

        images = preprocess_input(np.array(images))
        return ({"img_input":images, "ann_input":annots}, labels)
    
    # Credits: https://www.kaggle.com/code1110/mixup-cutmix-in-keras, with modification
    def mixup(self, X1, y1, X2, y2):
        assert X1.shape[0] == y1.shape[0] == X2.shape[0] == y2.shape[0]
        n_size = X1.shape[0]
        l = np.random.beta(self.mixup_alpha, self.mixup_alpha, n_size)
        X_l = l.reshape(n_size, 1, 1, 1)
        y_l = l.reshape(n_size, 1)
        X = X1 * X_l + X2 * (1-X_l)
        y = y1 * y_l + y2 * (1-y_l)
        return X, y
    
    def get_rand_bbox(self, width, height, l):
        r_x = np.random.randint(width)
        r_y = np.random.randint(height)
        r_l = np.sqrt(1 - l)
        r_w = np.int(width * r_l)
        r_h = np.int(height * r_l)
        return r_x, r_y, r_l, r_w, r_h
    
    def cutmix(self, X1, y1, X2, y2):
        assert X1.shape[0] == y1.shape[0] == X2.shape[0] == y2.shape[0]
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        width = X1.shape[1]
        height = X1.shape[0]
        r_x, r_y, r_l, r_w, r_h = self.get_rand_bbox(width, height, lam)
        bx1 = np.clip(r_x - r_w // 2, 0, width)
        by1 = np.clip(r_y - r_h // 2, 0, height)
        bx2 = np.clip(r_x + r_w // 2, 0, width)
        by2 = np.clip(r_y + r_h // 2, 0, height)
        X1[:, bx1:bx2, by1:by2, :] = X2[:, bx1:bx2, by1:by2, :]
        y = y1 * lam + y2 * (1-lam)
        return X1, y
    
    def augmentor(self, images):
        # Apply data augmentation with imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
                [
                # apply the following augmenters to most images
                iaa.Fliplr(p=0.5),  # horizontally flip 50% of all images
                sometimes([iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    # scale images to 90-110% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    # translate by -10 to +10 percent (per axis)
                    rotate=(-10, 10),  # rotate by -10 to +10 degrees
                    shear=(-5, 5),  # shear by -5 to +5 degrees
                    order=[0, 1],
                    # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )]),
                ],
                random_order=True
        )
        return seq.augment_images(images)
    
class TTADataGenerator(tf.keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, images_paths, ocr_annots, labels, n_class, batch_size=1, img_dim=(128,128,3), 
                 shuffle=False):
        self.labels       = labels       # array of labels
        self.n_class      = n_class      # number of class
        self.images_paths = images_paths # array of image paths
        self.ocr_annots   = ocr_annots
        self.dim          = img_dim      # image dimensions
        self.batch_size   = batch_size   # batch size
        self.shuffle      = shuffle      # shuffle bool
        self.choose       = 0            # switch for type of augmentation, zero value for no aug.
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # Generate one batch of data
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        
        # select ocr annots
        annots = self.ocr_annots[indexes]

        # select data and load images
        labels = self.labels[indexes]
        labels = tf.keras.utils.to_categorical(labels, num_classes=self.n_class)
        images = [cv2.imread(self.images_paths[k]) for k in indexes]
        images = [cv2.resize(img, self.dim[:2], cv2.INTER_AREA) for img in images]
        
        # preprocess and apply TTA
        if(self.choose!=0):
            images = self.augmentor(images)
        return ({"img_input":images, "ann_input":annots}, labels)

        images = preprocess_input(np.array(images))
        return images, labels
    
    def augmentor(self, images):
        # Define your list of augmentation for TTA here. This was designed to give deterministic output
        aug_list = [
                    iaa.Fliplr(p=1.0),
                    iaa.Affine(rotate=(-10, 10), mode='constant', cval=(0, 255), random_state=88),
                    iaa.Affine(translate_percent={"x":(-0.1, 0.1), "y":(-0.1, 0.1)}, mode='constant', cval=(0, 255), random_state=88),
                   ]
        # Choose augmentation
        seq = aug_list[self.choose-1]
        return seq.augment_images(images)

# LR schedule
def build_lrfn(lr_start=0.0001, lr_max=0.0001, 
               lr_min=0.00001, lr_rampup_epochs=0, 
               lr_sustain_epochs=0, lr_exp_decay=.8):

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    
    return lrfn


class NetModel():
    def __init__(self, img_dim=(96,96,3), ann_dim=(100, ), n_classes=1, n_epochs=10):
        self.n_classes = n_classes  # number of classes to classify(1 for binary classification)
        self.img_input_dim = img_dim  # image input dimensions
        self.ann_input_dim = ann_dim
        self.model = self.create_model()  # model
        self.n_epochs = n_epochs

    def summary(self):
        self.model.summary()

    def create_model(self):
        img_input_layer = tf.keras.layers.Input(self.img_input_dim, name="img_input")
        ann_input_layer = tf.keras.layers.Input(self.ann_input_dim, name="ann_input")
        
        ## Uncomment lines according to model type
        # If training Xception, use first line. If training InceptionResNetV2, use second line, etc.
        model = Xception(input_tensor=img_input_layer, include_top=False, weights='imagenet')
#         model = InceptionResNetV2(input_tensor=img_input_layer, include_top=False, weights='imagenet')
#         model = EfficientNetB3(input_tensor=img_input_layer, include_top=False, weights='imagenet')
        x = model(img_input_layer)
        
        # output layers
        x1 = tf.keras.layers.GlobalAveragePooling2D()(x)
        x2 = tf.keras.layers.GlobalMaxPooling2D()(x)
        x3 = tf.keras.layers.Flatten()(x)
        
        # Annot model
        embed = tf.keras.layers.Embedding(126964, 64, input_length=self.ann_input_dim[0])
        m = embed(ann_input_layer)
        m = tf.keras.layers.Flatten()(m)

        out = tf.keras.layers.Concatenate(axis=-1)([x1, x2, x3, m])
        out = tf.keras.layers.Dropout(0.3)(out)
        output_layer = tf.keras.layers.Dense(self.n_classes, activation='softmax')(out)

        model = tf.keras.models.Model(inputs=[img_input_layer, ann_input_layer], outputs=output_layer)

        model.compile(optimizer=tf.keras.optimizers.Adam(), 
                      loss="categorical_crossentropy", metrics=['acc'])
        return model


    def train(self, train_data, val_data, save_name):
        # Trains data on generators
        print("Starting training")

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(build_lrfn(), verbose=True)

        # stop training if no improvements are seen
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_acc",
                                                      mode="max",
                                                      patience=3,
                                                      restore_best_weights=True)

        # saves model weights to file
        checkpoint = tf.keras.callbacks.ModelCheckpoint('./weights_{}.hdf5'.format(save_name),
                                                        monitor='val_acc',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        mode='max',
                                                        save_weights_only=True)

        # train on data
        history = self.model.fit(x=train_data,
                                 validation_data=val_data,
                                 epochs=self.n_epochs,
                                 steps_per_epoch=len(train_data),
                                 validation_steps=len(val_data),
                                 callbacks=[lr_scheduler, checkpoint, early_stop], verbose=1,
                                 workers=4
                                )

    def prediction(self, tta_val_data, tta_tes_data, save_name, pred_tta=True, tta_round=5):
        # Create basic file submit
        self.model.load_weights("./weights_{}.hdf5".format(save_name))
        
        if(pred_tta):
            print('Predicting test data (TTA)..')
            for tta in range(tta_round):
                if(tta==0):
                    tes_pred = self.model.predict(x=tta_tes_data) / tta_round
                else:
                    tes_pred += self.model.predict(x=tta_tes_data) / tta_round
                tta_tes_data.choose += 1
            print(tes_pred.shape)
            print('  Saving to csv..')
            df = pd.DataFrame(tes_pred)
            df['filename'] = tta_tes_data.images_paths
            df["filename"] = df["filename"].apply(lambda x: x.split('/')[-1])
            df.to_csv("./tesPred_tta_{}.csv".format(save_name), index=False)
            del tes_pred, df
        
            print('Predicting validation data (TTA)..')
            for tta in range(tta_round):
                if(tta==0):
                    val_pred = self.model.predict(x=tta_val_data) / tta_round
                else:
                    val_pred += self.model.predict(x=tta_val_data) / tta_round
                tta_val_data.choose += 1
            print(val_pred.shape)
            print('  Saving to csv..')
            df = pd.DataFrame(val_pred)
            df['filename'] = tta_val_data.images_paths
            df["filename"] = df["filename"].apply(lambda x: x.split('/')[-1])
            df.to_csv("./valPred_tta_{}.csv".format(save_name), index=False)
            del val_pred, df
        print('Done')

def load_data(tra_df, val_df, tes_df, tra_ocr, val_ocr, tes_ocr, img_dim, augment_train=True, n_class=2, batch_size=64):
    X_tra = tra_df['path'].values
    y_tra = tra_df['category'].values
    X_val = val_df['path'].values
    y_val = val_df['category'].values
    X_tes = tes_df['path'].values
    y_tes = tes_df['category'].values
        
    tra_data = DataGenerator(X_tra, tra_ocr, y_tra, n_class=n_class, img_dim=img_dim, batch_size=batch_size, augment=augment_train, shuffle=True)
    val_data = DataGenerator(X_val, val_ocr, y_val, n_class=n_class, img_dim=img_dim, batch_size=batch_size, augment=False, shuffle=False)
    tta_val_data = TTADataGenerator(X_val, val_ocr, y_val, n_class=n_class, img_dim=img_dim, batch_size=1, shuffle=False)
    tta_tes_data = TTADataGenerator(X_tes, tes_ocr, y_tes, n_class=n_class, img_dim=img_dim, batch_size=1, shuffle=False)
        
    return tra_data, val_data, tta_val_data, tta_tes_data

## Load metadata
train_df = pd.read_csv('../input/shopee-product-detection-open/train.csv')
test_df = pd.read_csv('../input/shopee-product-detection-open/test.csv')

train_df['path'] = '../input/shopee-product-detection-open/train/train/train/'
train_df['path'] += train_df['category'].apply(lambda x: "{:02d}/".format(x))
train_df['path'] += train_df['filename']

test_df['path'] = '../input/shopee-product-detection-open/test/test/test/'
test_df['path'] += test_df['filename']
test_df['category'] = 0

X = train_df[['path']]
y = train_df[['category']]

tes_fold_df = test_df[['path', 'category']]

# Split data in 5 StratifiedKfold
tra_fold_df = []
val_fold_df = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=88)
for tra_idx, val_idx in skf.split(X, y):
    X_tra, y_tra = X.iloc[tra_idx], y.loc[tra_idx]
    X_val, y_val = X.iloc[val_idx], y.loc[val_idx]
    tra_fold_df.append(X_tra.join(y_tra))
    val_fold_df.append(X_val.join(y_val))

# Prepare OCR data
ocr_train = pd.read_csv('../input/scl-product-detection-useful-data/train_ocr.csv')
ocr_test = pd.read_csv('../input/scl-product-detection-useful-data/test_ocr.csv')

with (open("../input/scl-product-detection-useful-data/typocorr.pickle", "rb")) as openfile:
    typo = pickle.load(openfile)

ocr_train['path'] = ocr_train['path'].apply(lambda x: '/'.join(x.split('/')[-2:]))
ocr_train['path'] = '../input/shopee-code-league-2020-product-detection/resized/train/' + ocr_train['path']

ocr_test['path'] = ocr_test['path'].apply(lambda x: x.split('/')[-1])
ocr_test['path'] = '../input/shopee-code-league-2020-product-detection/resized/test/' + ocr_test['path']

ocr_tra_fold0 = pd.merge(tra_fold_df[0], ocr_train, on='path', how='left')
ocr_tra_fold0 = ocr_tra_fold0.fillna('')

ocr_val_fold0 = pd.merge(val_fold_df[0], ocr_train, on='path', how='left')
ocr_val_fold0 = ocr_val_fold0.fillna('')

ocr_tes = pd.merge(tes_fold_df, ocr_test, on='path', how='left')
ocr_tes = ocr_tes.fillna('')

ocr_tra_fold0['annot'] = ocr_tra_fold0['annot'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower())
ocr_tra_fold0['annot'] = ocr_tra_fold0['annot'].apply(lambda x: ' '.join(map(lambda s: typo.get(s, s), x.split())))

ocr_val_fold0['annot'] = ocr_val_fold0['annot'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower())
ocr_val_fold0['annot'] = ocr_val_fold0['annot'].apply(lambda x: ' '.join(map(lambda s: typo.get(s, s), x.split())))

ocr_tes['annot'] = ocr_tes['annot'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower())
ocr_tes['annot'] = ocr_tes['annot'].apply(lambda x: ' '.join(map(lambda s: typo.get(s, s), x.split())))

max_len = 100
vocab_size = 126964

all_ocr = ocr_tra_fold0['annot'].append(ocr_val_fold0['annot'])
all_ocr = all_ocr.append(ocr_tes['annot'])

ocr_seq = [tf.keras.preprocessing.text.one_hot(d, vocab_size, split=' ') for d in all_ocr]

ocr_tra_fold0_vec = ocr_seq[:len(ocr_tra_fold0)]
ocr_val_fold0_vec = ocr_seq[len(ocr_tra_fold0):len(ocr_tra_fold0)+len(ocr_val_fold0)]
ocr_tes_vec = ocr_seq[len(ocr_tra_fold0)+len(ocr_val_fold0):len(ocr_tra_fold0)+len(ocr_val_fold0)+len(ocr_tes)]

ocr_tra_fold0_vec = tf.keras.preprocessing.sequence.pad_sequences(ocr_tra_fold0_vec, maxlen=max_len, padding='pre')
ocr_val_fold0_vec = tf.keras.preprocessing.sequence.pad_sequences(ocr_val_fold0_vec, maxlen=max_len, padding='pre')
ocr_tes_vec = tf.keras.preprocessing.sequence.pad_sequences(ocr_tes_vec, maxlen=max_len, padding='pre')

# Simple configuration
EPOCHS = 13
BATCH_SIZE = 32
IMAGE_DIMENSIONS = (299,299,3)
ANNOT_DIMENSIONS = (100, )

## Use below lines according to model type
## If training Xception, use first line. If training InceptionResNetV2, use second line, etc.
SAVE_NAME = 'fold0_[Xception-299]_aug2_ocr'
# SAVE_NAME = 'fold0_[InceptionResNetV2-299]_aug2_ocr'
# SAVE_NAME = 'fold0_[EfficientNetB3-299]_aug2_ocr'

# Initialized model
model = NetModel(ann_dim=ANNOT_DIMENSIONS, img_dim=IMAGE_DIMENSIONS, n_classes=42, n_epochs=EPOCHS)

# Choose fold 0 split as training and validation
tra_data, val_data, tta_val_data, tta_tes_data = load_data(tra_fold_df[0], val_fold_df[0], tes_fold_df,
                                                           ocr_tra_fold0_vec, ocr_val_fold0_vec, ocr_tes_vec,
                                                           img_dim=IMAGE_DIMENSIONS, n_class=42, augment_train=True, 
                                                           batch_size=BATCH_SIZE)

model.train(tra_data, val_data, SAVE_NAME)
## NOTE:
## If the training takes longer than the kernel time limit, 
## run the training first (above lines), then do a test (below code) on another run, without training again
## make sure to add weights from previous output into input data
model.prediction(tta_val_data, tta_tes_data, SAVE_NAME, pred_tta=True)
