import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose, Lambda, \
    BatchNormalization, Bidirectional, LSTM, Dropout, Dense, InputLayer, Conv2D, MaxPooling2D, Flatten,\
    AveragePooling2D, GlobalAveragePooling2D, GlobalAveragePooling1D, AveragePooling1D, MultiHeadAttention,\
    LayerNormalization, Embedding, LeakyReLU, Conv1DTranspose


def cnn_model(max_len, vocab_size):
    model = Sequential([
        InputLayer(input_shape=(max_len, vocab_size)),
        Conv1D(32, 17, padding='same', activation='relu'),
        Conv1D(64, 11, padding='same', activation='relu'),
        Conv1D(128, 5, padding='same', activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    return model

## Step 1: Implement your own model below

## Step 2: Add your model name and model initialisation in the model dictionary below

def return_model(model_name, max_len, vocab_size):
    model_dic={'cnn': cnn_model(max_len, vocab_size),
               'se_cnn': se_cnn(max_len, vocab_size),
               'unet': cnn_model(max_len, vocab_size)}
    return model_dic[model_name]



class SqueezeExcitation1DLayer(tf.keras.Model):

    def __init__(self, out_dim, ratio, layer_name="se"):
        super(SqueezeExcitation1DLayer, self).__init__(name=layer_name)
        self.squeeze = GlobalAveragePooling1D()
        self.excitation_a = Dense(units=out_dim / ratio, activation='relu')
        self.excitation_b = Dense(units=out_dim, activation='sigmoid')
        self.shape = [-1, 1, out_dim]

    def call(self, input_x):
        squeeze = self.squeeze(input_x)

        excitation = self.excitation_a(squeeze)
        excitation = self.excitation_b(excitation)

        scale = tf.reshape(excitation, self.shape)
        se = input_x * scale

        return se
    
    def get_config(self):
        return {"shape": self.shape}


def se_cnn(max_len, vocab_size):
    model = Sequential([
        InputLayer(input_shape=(max_len, vocab_size)),
        Conv1D(32, 17, padding='same', activation='relu'),
        BatchNormalization(),
        LayerNormalization(),
        SqueezeExcitation1DLayer(out_dim=32, ratio=2, layer_name='se0'),
        Dropout(0.5),
        Conv1D(64, 11, padding='same', activation='relu'),
        BatchNormalization(),
        LayerNormalization(),
        SqueezeExcitation1DLayer(out_dim=64, ratio=4, layer_name='se1'),
        Dropout(0.5),
        Conv1D(128, 5, padding='same', activation='relu'),
        BatchNormalization(),
        LayerNormalization(),
        SqueezeExcitation1DLayer(out_dim=128, ratio=8, layer_name='se2'),
        Conv1D(128, 3, padding='same', activation='relu'),
        SqueezeExcitation1DLayer(out_dim=128, ratio=8),
        Dense(1, activation='sigmoid')
    ])
    return model

def unet(max_len, vocab_size):
    model = UNet(input_size=(max_len, vocab_size))
    return model


# UNet adopted from https://github.com/VidushiBhatia/U-Net-Implementation


def EncoderBlock(inputs, n_filters=15, kernel_size=7, dropout_prob=0.3, max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning.
    Dropout can be added for regularization to prevent overfitting.
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """
    # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow
    # Proper initialization prevents from the problem of exploding and vanishing gradients
    # 'Same' padding will pad the input to conv layer such that the output has the same height and width
    # (hence, is not reduced in size)
    conv = Conv1D(n_filters,
                  kernel_size,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    bn = BatchNormalization()(conv)
    conv = Conv1D(n_filters,
                  kernel_size,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(bn)

    # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
    conv = BatchNormalization()(conv, training=False)

    # In case of overfitting, dropout will regularize the loss and gradient computation to shrink
    # the influence of weights on output
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # Pooling reduces the size of the image while keeping the number of channels same Pooling has been kept as
    # optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use) Below,
    # Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse
    # across input image
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during
    # transpose convolutions
    skip_connection = conv

    return next_layer, skip_connection


def DecoderBlock(prev_layer_input, skip_layer_input, kernel_size=7, n_filters=32):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    # Start with a transpose convolution layer to first increase the size of the image
    up = Conv1DTranspose(#input_tensor=prev_layer_input,
                         filters=n_filters,
                         kernel_size=kernel_size,  # Kernel size
                         strides=2,
                         padding='same')(prev_layer_input)

    # Merge the skip connection from previous block to prevent information loss
    merge = concatenate([up, skip_layer_input], axis=-1)

    # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
    # The parameters for the function are similar to encoder
    conv = Conv1D(n_filters,
                  kernel_size,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(merge)
    bn = BatchNormalization()(conv)
    conv = Conv1D(n_filters,
                  kernel_size,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(bn)
    bn = BatchNormalization()(conv)
    return bn


def UNet(input_size=(10240, 5), n_filters=32, n_classes=1):
    """
    Combine both encoder and decoder blocks according to the U-Net research paper
    Return the model as output
    """
    # Input size represent the size of 1 image (the size used for pre-processing)
    inputs = Input(input_size)

    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of
    # the image
    cblock1 = EncoderBlock(inputs, n_filters, dropout_prob=0, max_pooling=True)
    cblock2 = EncoderBlock(cblock1[0], 22, dropout_prob=0, max_pooling=True)
    cblock3 = EncoderBlock(cblock2[0], 33, dropout_prob=0, max_pooling=True)
    cblock4 = EncoderBlock(cblock3[0], 49, dropout_prob=0.3, max_pooling=True)
    cblock5 = EncoderBlock(cblock4[0], 73, dropout_prob=0.3, max_pooling=False)

    # Decoder includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the encoder are given as input to the decoder
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
    ublock6 = DecoderBlock(cblock5[0], cblock4[1], 109)
    ublock7 = DecoderBlock(ublock6, cblock3[1], 72)
    ublock8 = DecoderBlock(ublock7, cblock2[1], 48)
    ublock9 = DecoderBlock(ublock8, cblock1[1], 32)

    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size.
    # Observe the number of channels will be equal to number of output classes
    conv9 = Conv1D(n_filters,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(ublock9)

    conv10 = Conv1D(n_classes, 1, activation='sigmoid', padding='same')(conv9)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model

