import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


# Convolutional block: 4 blocks on decoder side, 4 blocks on encoder side and 1 bridge
def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # each block based on 2 layers
    return x


# vuild the model


def build_model(shape):
    num_filters = [64, 128, 256, 512]  # no. neurons in each of 4 building blocks
    inputs = Input((shape))  # shape of data

    skip_x = []
    x = inputs

    # Encoder - model begins with input layer, iterate through list of filters, progressively increasing number of filters in each convolutional block
    for f in num_filters:
        x = conv_block(x, f)
        # for each filtered [crown] apply conv_block function, which includes 2 confolution layeres, batch normalisation, and reulu activation
        skip_x.append(x)  # track intermediate result in this list
        x = MaxPool2D((2, 2))(x)
        # downsample the feature maps using the max pooling layers

    # Bridge (bottleneck layer)

    x = conv_block(x, 1024)
    # another conv. layer with 1024 layers to capture high-level features

    num_filters.reverse()
    skip_x.reverse()
    # reverse order of filtered crown and feature maps

    # Decoder
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2))(x)  # upsamples the feature maps
        xs = skip_x[i]
        x = Concatenate()([x, xs])  # concatenate into corresponding featuremaps
        x = conv_block(
            x, f
        )  # apply to each concatenated feature map to decode and refine the segmentation info

    # output layer
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)  # this is a binary model

    return Model(inputs, x)


############

import numpy as np

# load the saved Numpy arrayes (train and test data)

print("Load the Train and Test Data :")
allImagesNP = np.load("C:/Data-Sets/temp/Unet-Train-Melanoa-Images.npy")
maskImagesNP = np.load("C:/Data-Sets/temp/Unet-Train-Melanoa-Masks.npy")
allTestImagesNP = np.load("C:/Data-Sets/temp/Unet-Test-Melanoa-Images.npy")
maskTestImagesNP = np.load("C:/Data-Sets/temp/Unet-Test-Melanoa-Masks.npy")


print(allImagesNP.shape)
print(maskImagesNP.shape)
print(allTestImagesNP.shape)
print(maskTestImagesNP.shape)

Height = 128
Width = 128

# build the model

# import tensorflow as tf
# from model import build_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

shape = (128, 128, 3)

lr = 1e-4  # 0.001
batch_size = 8
epochs = 20

model = build_model(shape)
print(model.summary())

opt = tf.keras.optimizers.Adam(lr)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

stepsPerEpoch = int(np.ceil(len(allImagesNP) / batch_size))  # round up the result
validationSteps = int(np.ceil(len(allTestImagesNP) / batch_size))  # round up the result

best_model_file = "C:/Data-Sets/temp/melanoma-Unet.keras"

# special functions in keras that can be called during training
# allows monitoring / optimizing of training process (chainging vlaues during chaining)
# Reduce LR allows to reduce learning rate on plateau
# Earlystopping callback stops after 20 epochs if no improvement to avoid overfitting
callbacks = [
    ModelCheckpoint(best_model_file, verbose=1, save_best_only=True),
    ReduceLROnPlateau(
        monitor="val_loss", patience=5, factor=0.1, verbose=1, min_lr=1e-7
    ),
    EarlyStopping(monitor="val_accuracy", patience=10, verbose=1),
]

history = model.fit(
    allImagesNP,
    maskImagesNP,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(allTestImagesNP, maskTestImagesNP),
    steps_per_epoch=stepsPerEpoch,
    validation_steps=validationSteps,
    shuffle=True,
    callbacks=callbacks,
)
