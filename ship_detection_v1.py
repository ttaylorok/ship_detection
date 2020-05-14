PATH = "C:\\Users\\comp\\Documents\\ship_detection\\airbus-ship-detection\\"
TRAIN_PATH = PATH + "train_v2\\"
TEST_PATH = PATH + "test_v2\\"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


masks = pd.read_csv(PATH + "train_ship_segmentations_v2.csv")

#%% check decode
test_mask = masks.loc[masks['ImageId'] == '000155de5.jpg']['EncodedPixels']
decode = rle_decode(test_mask.iloc[0])
test_img = skimage.io.imread(PATH + 'train_v2\\000155de5.jpg')
plt.imshow(test_img)
plt.imshow(decode, alpha = 0.4)

#%% create simple generator

def image_gen(df, show = False):
    images = list(df['ImageId'])
    masks = list(df['EncodedPixels'])
    i = 0
    while i < len(images):
        img = skimage.io.imread(TRAIN_PATH + images[i])
        mask = np.reshape(rle_decode(masks[i]),(768,768,1))
        if show == True:
            plt.imshow(img)
            plt.imshow(mask, alpha = 0.4)
        yield (np.array([img/255]), np.array([mask]))
        i += 1
        


#%% create small sample for testing

from sklearn.model_selection import train_test_split

seed = 21
mso = masks.dropna()
mso['count'] = mso.groupby('ImageId').transform('count')
mso_filt = mso[mso['count'] == 1].iloc[0:100]

# X = []
# y = []
# for row in mso_filt.iterrows():
#     X.append(skimage.io.imread(PATH + 'train_v2//' + row[1][0]))
#     tt = rle_decode(row[1][1])
#     tt2 = np.reshape(tt,(768,768,1))
#     y.append(tt2)

df_train, df_test = train_test_split(mso_filt, test_size = 0.5, shuffle=True)

train_gen = image_gen(df_train)
X_train, y_train = next(train_gen)
print(np.shape(X_train), np.shape(y_train))

test_gen = image_gen(df_test)
X_test, y_test = next(test_gen)

#%% create model
#https://towardsdatascience.com/a-keras-pipeline-for-image-segmentation-part-1-6515a421157d
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import model

#in_size = (1,768,768,3)
m = model.unet(input_size = np.shape(X_train)[1:])
opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

BATCH_SIZE = 1

results = m.fit_generator(image_gen(df_train),
                          epochs = 5, 
                          steps_per_epoch = 10,
                          validation_data = image_gen(df_test), 
                          validation_steps = 10)
m.save('Model.h5')


    
# X_train, X_test, y_train, y_test = train_test_split(
#     np.array(X), np.array(y), test_size=0.5, random_state=42)

#%% subset and prepare data

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils


from tensorflow import keras
from tensorflow.keras import layers

# Set random seed for purposes of reproducibility
seed = 21

mso = masks.dropna()

mso['count'] = mso.groupby('ImageId').transform('count')

mso_filt = mso[mso['count'] == 1].iloc[0:100]

X = []
y = []
for row in mso_filt.iterrows():
    X.append(skimage.io.imread(PATH + 'train_v2//' + row[1][0]))
    tt = rle_decode(row[1][1])
    tt2 = np.reshape(tt,(768,768,1))
    y.append(tt2)
    
X_train, X_test, y_train, y_test = train_test_split(
    np.array(X), np.array(y), test_size=0.5, random_state=42)
    
#%% model
    
# tt = rle_decode(row[1][1])
# np.shape(tt)
# tt2 = np.reshape(tt,(768,768,1))
# np.shape(tt2)




input_shape = X_train.shape[1:]
num_classes = 2
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
print(model.summary())

batch_size = 128
epochs = 15

import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_p_bce(in_gt, in_pred):
    return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)

#model.compile(loss=dice_p_bce, optimizer="adam", metrics=["accuracy"])
#model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=dice_p_bce, metrics=[dice_coef, 'binary_accuracy', true_positive_rate])
model.compile(loss="crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

from keras import Input

img_input = Input(shape=np.shape(X_train)[1:])

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
conv1 = Dropout(0.2)(conv1)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D((2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Dropout(0.2)(conv2)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D((2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Dropout(0.2)(conv3)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

up1 = np.concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
conv4 = Dropout(0.2)(conv4)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

up2 = np.concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
conv5 = Dropout(0.2)(conv5)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

out = Conv2D( n_classes, (1, 1) , padding='same')(conv5)

from keras_segmentation.models.model_utils import get_segmentation_model

model = get_segmentation_model(img_input ,  out ) # this would build the segmentation model

#%% from https://segmentation-models.readthedocs.io/en/latest/tutorial.html

from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)

# load your data
#x_train, y_train, x_val, y_val = load_data(...)

# preprocess input
X_train_pp = preprocess_input(X_train)
X_test_pp = preprocess_input(X_test)

# define model
model = Unet(BACKBONE, encoder_weights='imagenet')
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

# fit model
model.fit(
    x=X_train_pp,
    y=y_train,
    batch_size=16,
    epochs=100,
    validation_data=(X_test_pp, y_test),
)






