import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, BatchNormalization

def initialize_with_pretrained_weights(model, weights_path):
    '''
    Initializes the weights of a model using those saved at weights_path
    :param model: A tf.keras model
    :param weights_path: A path to serialized model weights
    :return: A tf.keras model with weights initialized using pre-trained values
    '''
    pretrained = tf.keras.models.load_model(weights_path, compile=False)
    for layer in pretrained.layers:
        if layer.trainable:
            model.get_layer(layer.name).set_weights(layer.get_weights())
    return model


# Skip Connector for custom ResNetV2
def residual_block(X, num_filters: int, stride: int = 1, kernel_size: int = 3,
                   activation: str = 'relu', bn: bool = True, conv_first: bool = True):
    """
    :param X: Tensor layer from previous layer
    :param num_filers: integer, conv2d number of filters
    :param stride: integer, default 1, stride square dimension
    :param kernel_size: integer, default 3, conv2d square kernel dimensions
    :param activation: string, default 'relu', activation function
    :param bn: bool, default True, to use Batch Normalization
    :param conv_first: bool, default True, conv-bn-activation (True) or bn-activation-conv (False)
    """

    conv_layer = Conv2D(num_filters,
                        kernel_size=kernel_size,
                        strides=stride,
                        padding='same')
    # X = input
    if conv_first:
        X = conv_layer(X)
        if bn:
            X = BatchNormalization()(X)
        if activation is not None:
            X = Activation(activation)(X)

    else:
        if bn:
            X = BatchNormalization()(X)
        if activation is not None:
            X = Activation(activation)(X)
        X = conv_layer(X)

    return X