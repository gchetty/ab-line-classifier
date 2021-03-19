import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Input, LeakyReLU, BatchNormalization,                                    
                                    Activation, Add, GlobalAveragePooling2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import Constant
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications import EfficientNetB7

def mobilenetv2(model_config, input_shape, metrics, mixed_precision=False, output_bias=None):
    '''
    Defines a model based on a pretrained MobileNetV2 for binary US classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param mixed_precision: Whether to use mixed precision (use if you have GPU with compute capacity >= 7.0)
    :param output_bias: bias initializer of output layer
    :return: a Keras Model object with the architecture defined in this method
    '''

    # Set hyperparameters
    '''
    ADD HYPERPARAMETERS HERE
    '''

    print("MODEL CONFIG: ", model_config)
    
    if mixed_precision:
        tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    if output_bias is not None:
        output_bias = Constant(output_bias)     # Set initial output bias

    # Start with pretrained MobileNetV2
    X_input = Input(input_shape, name='input')
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)
    
    # Freeze layers
    '''
    ADD FROZEN LAYERS HERE
    '''

    X = base_model.output

    # Add custom top layers
    '''
    ADD CUSTOM TOP LAYERS HERE
    '''

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model
    
def vgg16(model_config, input_shape, metrics, mixed_precision=False, output_bias=None):
    '''
    Defines a model based on a pretrained VGG16 for binary US classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param mixed_precision: Whether to use mixed precision (use if you have GPU with compute capacity >= 7.0)
    :param output_bias: bias initializer of output layer
    :return: a Keras Model object with the architecture defined in this method
    '''

    # Set hyperparameters
    '''
    ADD HYPERPARAMETERS HERE
    '''

    print("MODEL CONFIG: ", model_config)
    
    if mixed_precision:
        tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    if output_bias is not None:
        output_bias = Constant(output_bias)     # Set initial output bias

    # Start with pretrained VGG16
    X_input = Input(input_shape, name='input')
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)
    
    # Freeze layers
    '''
    ADD FROZEN LAYERS HERE
    '''
    
    X = base_model.output

    # Add custom top layers
    '''
    ADD CUSTOM TOP LAYERS HERE
    '''

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model

def xception(model_config, input_shape, metrics, mixed_precision=False, output_bias=None):
    '''
    Defines a model based on a pretrained Xception for bianry US classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param mixed_precision: Whether to use mixed precision (use if you have GPU with compute capacity >= 7.0)
    :param output_bias: bias initializer of output layer
    :return: a Keras Model object with the architecture defined in this method
    '''

     # Set hyperparameters
    '''
    ADD HYPERPARAMETERS HERE
    '''

    print("MODEL CONFIG: ", model_config)
    
    if mixed_precision:
        tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    if output_bias is not None:
        output_bias = Constant(output_bias)     # Set initial output bias

    # Start with pretrained Xception
    X_input = Input(input_shape, name='input')
    base_model = Xception(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)
    
    # Freeze layers
    '''
    ADD FROZEN LAYERS HERE
    '''
    
    X = base_model.output

    # Add custom top layers
    '''
    ADD CUSTOM TOP LAYERS HERE
    '''

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model
    
def efficientnetb7(model_config, input_shape, metrics, mixed_precision=False, output_bias=None):
    '''
     Defines a model based on a pretrained EfficientNetB7 for binary US classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param mixed_precision: Whether to use mixed precision (use if you have GPU with compute capacity >= 7.0)
    :param output_bias: bias initializer of output layer
    :return: a Keras Model object with the architecture defined in this method
    '''

    # Set hyperparameters
    '''
    ADD HYPERPARAMETERS HERE
    '''

    print("MODEL CONFIG: ", model_config)
    
    if mixed_precision:
        tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    if output_bias is not None:
        output_bias = Constant(output_bias)     # Set initial output bias

    # Pre-trained architecture
    X_input = Input(input_shape, name='input')
    base_model = EfficientNetB7(weights='imagenet', input_shape=input_shape, include_top=False, input_tensor=X_input)

    # Freeze layers
    '''
    ADD FROZEN LAYERS HERE
    '''

    X = base_model.output

    # Add custom top layers
    '''
    ADD CUSTOM TOP LAYERS HERE
    '''

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model