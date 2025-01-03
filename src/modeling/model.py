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
        x = conv_block(
            x, f
        )  # for each filtered [crown] apply conv_block function, which includes 2 confolution layeres, batch normalisation, and reulu activation
        skip_x.append(x)  # track intermediate result in this list
        x = MaxPool2D((2, 2))(
            x
        )  # downsample the feature maps using the max pooling layers

    # Bridge (bottleneck layer)

    x = conv_block(
        x, 1024
    )  # another conv. layer with 1024 layers to capture high-level features

    num_filters.reverse()
    skip_x.reverse()  # reverse order of filtered crown and feature maps

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
