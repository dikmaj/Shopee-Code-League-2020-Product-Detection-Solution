"""
This script is used to trained Xception, InceptionResNetV2, EfficientNet B3 and B5 on 299x299 image. 
Simple augmentation (see below) is used for training and 
Test Time Augmentation (TTA) scheme is used for validation and test data

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

## Uncomment lines according to model type
## If training Xception, use first line. If training InceptionResNetV2, use second line, etc.
from tensorflow.keras.applications.xception import Xception, preprocess_input
# from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
# from efficientnet.tfkeras import EfficientNetB3, preprocess_input
# from efficientnet.tfkeras import EfficientNetB5, preprocess_input

# Credits: https://www.kaggle.com/mpalermo/keras-pipeline-custom-generator-imgaug
class DataGenerator(tf.keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, images_paths, labels, n_class, batch_size=64, img_dim=(128,128,3), 
                 shuffle=False, augment=False):
        self.labels       = labels       # array of labels
        self.n_class      = n_class      # number of class
        self.images_paths = images_paths # array of image paths
        self.dim          = img_dim      # image dimensions
        self.batch_size   = batch_size   # batch size
        self.shuffle      = shuffle      # shuffle bool
        self.augment      = augment      # augment data bool
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

        # select data and load images
        labels = self.labels[indexes]
        labels = tf.keras.utils.to_categorical(labels, num_classes=self.n_class)
        images = [cv2.imread(self.images_paths[k]) for k in indexes]
        images = [cv2.resize(img, self.dim[:2], cv2.INTER_AREA) for img in images]
        
        # preprocess and augment data
        if self.augment == True:
            images = self.augmentor(images)

        images = preprocess_input(np.array(images))
        return images, labels
    
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
    def __init__(self, images_paths, labels, n_class, batch_size=1, img_dim=(128,128,3), 
                 shuffle=False):
        self.labels       = labels       # array of labels
        self.n_class      = n_class      # number of class
        self.images_paths = images_paths # array of image paths
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

        # select data and load images
        labels = self.labels[indexes]
        labels = tf.keras.utils.to_categorical(labels, num_classes=self.n_class)
        images = [cv2.imread(self.images_paths[k]) for k in indexes]
        images = [cv2.resize(img, self.dim[:2], cv2.INTER_AREA) for img in images]
        
        # preprocess and apply TTA
        if(self.choose!=0):
            images = self.augmentor(images)

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
    def __init__(self, img_dim=(96,96,3), n_classes=1, n_epochs=10):
        self.n_classes = n_classes  # number of classes to classify (1 for binary classification)
        self.input_dim = img_dim  # image input dimensions
        self.model = self.create_model()  # model
        self.n_epochs = n_epochs

    def summary(self):
        self.model.summary()

    def create_model(self):
        input_layer = tf.keras.layers.Input(self.input_dim)
        
        ## Uncomment lines according to model type
        # If training Xception, use first line. If training InceptionResNetV2, use second line, etc.
        model = Xception(input_tensor=input_layer, include_top=False, weights='imagenet')
#         model = InceptionResNetV2(input_tensor=input_layer, include_top=False, weights='imagenet')
#         model = EfficientNetB3(input_tensor=input_layer, include_top=False, weights='imagenet')
#         model = EfficientNetB5(input_tensor=input_layer, include_top=False, weights='imagenet')
        x = model(input_layer)

        # output layers
        x1 = tf.keras.layers.GlobalAveragePooling2D()(x)
        x2 = tf.keras.layers.GlobalMaxPooling2D()(x)
        x3 = tf.keras.layers.Flatten()(x)

        out = tf.keras.layers.Concatenate(axis=-1)([x1, x2, x3])
        out = tf.keras.layers.Dropout(0.3)(out)
        output_layer = tf.keras.layers.Dense(self.n_classes, activation='softmax')(out)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

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

def load_data(tra_df, val_df, tes_df, img_dim, augment_train=True, n_class=2, batch_size=64):
    X_tra = tra_df['path'].values
    y_tra = tra_df['category'].values
    X_val = val_df['path'].values
    y_val = val_df['category'].values
    X_tes = tes_df['path'].values
    y_tes = tes_df['category'].values
        
    tra_data = DataGenerator(X_tra, y_tra, n_class=n_class, img_dim=img_dim, batch_size=batch_size, augment=augment_train, shuffle=True)
    val_data = DataGenerator(X_val, y_val, n_class=n_class, img_dim=img_dim, batch_size=batch_size, augment=False, shuffle=False)
    tta_val_data = TTADataGenerator(X_val, y_val, n_class=n_class, img_dim=img_dim, batch_size=1, shuffle=False)
    tta_tes_data = TTADataGenerator(X_tes, y_tes, n_class=n_class, img_dim=img_dim, batch_size=1, shuffle=False)
        
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

# Simple configuration
EPOCHS = 13
BATCH_SIZE = 32
IMAGE_DIMENSIONS = (299,299,3)

## Use below lines according to model type
## If training Xception, use first line. If training InceptionResNetV2, use second line, etc.
SAVE_NAME = 'fold0_[Xception-299]_aug'
# SAVE_NAME = 'fold0_[InceptionResNetV2-299]_aug'
# SAVE_NAME = 'fold0_[EfficientNetB3-299]_aug'
# SAVE_NAME = 'fold0_[EfficientNetB5-299]_aug'

# Initialized model
model = NetModel(img_dim=IMAGE_DIMENSIONS, n_classes=42, n_epochs=EPOCHS)

# Choose fold 0 split as training and validation
tra_data, val_data, tta_val_data, tta_tes_data = load_data(tra_fold_df[0], val_fold_df[0], tes_fold_df, 
                                                           img_dim=IMAGE_DIMENSIONS, n_class=42, augment_train=True, 
                                                           batch_size=BATCH_SIZE)

model.train(tra_data, val_data, SAVE_NAME)

## NOTE:
## If the training takes longer than the kernel time limit, 
## run the training first (above lines), then do a test (below code) on another run, without training again
## make sure to add weights from previous output into input data
model.prediction(tta_val_data, tta_tes_data, SAVE_NAME, pred_tta=True)
