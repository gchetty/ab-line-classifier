import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, GlobalAveragePooling2D, Conv2D, MaxPool2D, \
    BatchNormalization, ZeroPadding2D, SpatialDropout2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import Constant
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocess

def get_model(model_name):
    '''
    Return the model definition and associated preprocessing function as specified in the config file
    :return: (TF model definition function, preprocessing function)
    '''

    if model_name == 'efficientnetb7':
        model_def = efficientnetb7
        preprocessing_function = efficientnet_preprocess
    elif model_name == 'vgg16':
        model_def = vgg16
        preprocessing_function = vgg16_preprocess
    elif model_name == 'mobilenetv2':
        model_def = mobilenetv2
        preprocessing_function = mobilenetv2_preprocess
    elif model_name == 'xception':
        model_def = xception
        preprocessing_function = xception_preprocess
    elif model_name == 'custom_resnetv2':
        model_def = custom_resnetv2
        preprocessing_function = resnetv2_preprocess
    elif model_name == 'cutoffvgg16':
        model_def = CutoffVGG16
        preprocessing_function = vgg16_preprocess
    else:
        model_def = cnn0
        preprocessing_function = mobilenetv2_preprocess
    return model_def, preprocessing_function

def mobilenetv2(model_config, input_shape, metrics, n_classes, mixed_precision=False, output_bias=None):
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
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    weight_decay = model_config['L2_LAMBDA']
    optimizer = Adam(learning_rate=lr)
    fc0_nodes = model_config['NODES_DENSE0']
    frozen_layers = model_config['FROZEN_LAYERS']

    print("MODEL CONFIG: ", model_config)
    
    if mixed_precision:
        tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    if output_bias is not None:
        output_bias = Constant(output_bias)     # Set initial output bias

    # Start with pretrained MobileNetV2
    X_input = Input(input_shape, name='input')
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input, alpha=0.75)
    
    # Freeze layers
    for layers in range(len(frozen_layers)):
        layer2freeze = frozen_layers[layers]
        print('Freezing layer: ' + str(layer2freeze))
        base_model.layers[layer2freeze].trainable = False

    X = base_model.output

    # Add custom top layers
    X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout)(X)
    #X = Dense(fc0_nodes, activation='relu', activity_regularizer=l2(weight_decay), name='fc0')(X)
    #X = Dropout(dropout)(X)
    X = Dense(n_classes, bias_initializer=output_bias, name='logits')(X)
    Y = Activation('softmax', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model
    
def vgg16(model_config, input_shape, metrics, n_classes, mixed_precision, output_bias=None):
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
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    weight_decay = model_config['L2_LAMBDA']
    optimizer = Adam(learning_rate=lr)
    frozen_layers = model_config['FROZEN_LAYERS']
    fc0_nodes = model_config['NODES_DENSE0']

    print("MODEL CONFIG: ", model_config)
    
    if mixed_precision:
        tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    if output_bias is not None:
        output_bias = Constant(output_bias)     # Set initial output bias

    # Start with pretrained VGG16
    X_input = Input(input_shape, name='input')
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)
    
    # Freeze layers
    for layers in range(len(frozen_layers)):
        layer2freeze = frozen_layers[layers]
        print('Freezing layer: ' + str(layer2freeze))
        base_model.layers[layer2freeze].trainable = False
    
    X = base_model.output

    # Add custom top layers
    X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout)(X)
    X = Dense(n_classes, bias_initializer=output_bias, name='logits')(X)
    Y = Activation('softmax', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model

def xception(model_config, input_shape, metrics, n_classes, mixed_precision, output_bias=None):
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
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    l2_lambda = model_config['L2_LAMBDA']
    optimizer = Adam(learning_rate=lr)
    frozen_layers = model_config['FROZEN_LAYERS']

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
    X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout)(X)
    X = Dense(n_classes, bias_initializer=output_bias, name='logits')(X)
    Y = Activation('softmax', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model
    
def efficientnetb7(model_config, input_shape, metrics, n_classes, mixed_precision=False, output_bias=None):
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
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    l2_lambda = model_config['L2_LAMBDA']
    optimizer = Adam(learning_rate=lr)
    frozen_layers = model_config['FROZEN_LAYERS']

    print("MODEL CONFIG: ", model_config)
    
    if mixed_precision:
        tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    if output_bias is not None:
        output_bias = Constant(output_bias)     # Set initial output bias

    # Pre-trained architecture
    X_input = Input(input_shape, name='input')
    base_model = EfficientNetB7(weights='imagenet', input_shape=input_shape, include_top=False, input_tensor=X_input)

    # Freeze layers
    # for layers in range(len(frozen_layers)):
    #     layer2freeze = frozen_layers[layers]
    #     print('Freezing layer: ' + str(layer2freeze))
    #     base_model.layers[layer2freeze].trainable = False

    X = base_model.output

    # Add custom top layers
    X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout)(X)
    X = Dense(n_classes, bias_initializer=output_bias, name='logits')(X)
    Y = Activation('softmax', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model


def cnn0(model_config, input_shape, metrics, n_classes, mixed_precision=False, output_bias=None):
    # Set hyperparameters
    nodes_dense0 = model_config['NODES_DENSE0']
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    l2_lambda = model_config['L2_LAMBDA']
    optimizer = Adam(learning_rate=lr)
    init_filters = model_config['INIT_FILTERS']
    filter_exp_base = model_config['FILTER_EXP_BASE']
    n_blocks = model_config['BLOCKS']
    kernel_size = (model_config['KERNEL_SIZE'], model_config['KERNEL_SIZE'])
    max_pool_size = (model_config['MAXPOOL_SIZE'], model_config['MAXPOOL_SIZE'])
    strides = (model_config['STRIDES'], model_config['STRIDES'])
    pad = kernel_size[0] // 2
    print("MODEL CONFIG: ", model_config)
    if mixed_precision:
        tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    if output_bias is not None:
        output_bias = Constant(output_bias)  # Set initial output bias

    # Input layer
    X_input = Input(input_shape)
    X = X_input
    X = ZeroPadding2D((pad, pad))(X)

    # Add blocks of convolutions and max pooling
    for i in range(n_blocks):
        filters = init_filters * (2 ** i)
        X = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', name='conv2d_block' + str(i) + '_0',
                   kernel_initializer='he_uniform', activation='relu', activity_regularizer=l2(l2_lambda))(X)
        X = BatchNormalization(axis=3, name='bn_block' + str(i))(X)
        if i < n_blocks - 1:
            X = MaxPool2D(max_pool_size, padding='same', name='maxpool' + str(i))(X)

    # Model head
    X = GlobalAveragePooling2D(name='global_avgpool')(X)
    X = Dropout(dropout)(X)
    X = Dense(nodes_dense0, kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda), activation='relu', name='fc0')(X)
    X = Dense(n_classes, bias_initializer=output_bias)(X)
    Y = Activation('softmax', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model

# Skip Connector for custom ResNetV2
def residual_block(model_config, X, num_filters: int, stride: int = 1, kernel_size: int = 3,
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
    dropout0 = model_config['DROPOUT0']

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
            X = SpatialDropout2D(dropout0)(X)

    else:
        if bn:
            X = BatchNormalization()(X)
        if activation is not None:
            X = Activation(activation)(X)
        X = conv_layer(X)

    return X

def custom_resnetv2(model_config, input_shape, metrics, n_classes, mixed_precision=False, output_bias=None):
    '''
     Defines a model based on a custom ResNetV2 for binary US classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param mixed_precision: Whether to use mixed precision (use if you have GPU with compute capacity >= 7.0)
    :param output_bias: bias initializer of output layer
    :return: a Keras Model object with the architecture defined in this method
    '''

    # Set hyperparameters
    print(input_shape)
    lr = model_config['LR']
    dropout1 = model_config['DROPOUT1']
    optimizer = Adam(learning_rate=lr)
    num_filters_in = model_config['INIT_FILTERS']
    stride = (model_config['STRIDES'], model_config['STRIDES'])
    num_res_block = model_config['BLOCKS']
    print("MODEL CONFIG: ", model_config)

    X_input = Input(shape=input_shape)

    # ResNet V2 performs Conv2D on X before spiting into two path
    X = residual_block(model_config=model_config, X=X_input, num_filters=num_filters_in, conv_first=True)

    # Building stack of residual units
    for stage in range(3):
        for unit_res_block in range(num_res_block):
            activation = 'relu'
            bn = True
            stride = 1
            # First layer and first stage
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if unit_res_block == 0:
                    activation = None
                    bn = False
                # First layer but not first stage
            else:
                num_filters_out = num_filters_in * 2
                if unit_res_block == 0:
                    stride = 2

            # bottleneck residual unit
            y = residual_block(model_config=model_config,
                               X=X,
                               num_filters=num_filters_in,
                               kernel_size=1,
                               stride=stride,
                               activation=activation,
                               bn=bn,
                               conv_first=False)
            y = residual_block(model_config=model_config,
                               X=y,
                               num_filters=num_filters_in,
                               conv_first=False)
            y = residual_block(model_config=model_config,
                               X=y,
                               num_filters=num_filters_out,
                               kernel_size=1,
                               conv_first=False)
            if unit_res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                X = residual_block(model_config=model_config,
                                   X=X,
                                   num_filters=num_filters_out,
                                   kernel_size=1,
                                   stride=stride,
                                   activation=None,
                                   bn=False)
            X = tf.keras.layers.add([X, y])
        num_filters_in = num_filters_out

    # Model head
    X = SpatialDropout2D(dropout1)(X)
    X = GlobalAveragePooling2D(name='global_avgpool')(X)
    #X = Dense(nodes_dense0, kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda), activation='relu',
              #name='fc0')(X)
    Y = Dense(n_classes, activation='softmax', dtype='float32', name='output', bias_initializer=output_bias)(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics, run_eagerly=True)
    return model


class CutoffVGG16:

    def __init__(self, model_config, input_shape, metrics, n_classes, mixed_precision=False, output_bias=None):
        self.lr_extract = model_config['LR_EXTRACT']
        self.lr_finetune = model_config['LR_FINETUNE']
        self.dropout = model_config['DROPOUT']
        self.cutoff_layer = model_config['CUTOFF_LAYER']
        self.finetune_layer = model_config['FINETUNE_LAYER']
        self.extract_epochs = model_config['EXTRACT_EPOCHS']
        self.optimizer_extract = Adam(learning_rate=self.lr_extract)
        self.optimizer_finetune = RMSprop(learning_rate=self.lr_finetune)
        self.output_bias = output_bias
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.metrics = metrics
        self.mixed_precision = mixed_precision
        self.model = self.define_model()

    def define_model(self):
        X_input = Input(shape=self.input_shape, name='input')
        vgg16 = VGG16(input_shape=self.input_shape, include_top=False, weights='imagenet')
        self.vgg16_layers = vgg16.layers[1:self.cutoff_layer]
        X = X_input
        for layer in self.vgg16_layers:
            X = layer(X)
        X = GlobalAveragePooling2D(name='global_avgpool')(X)
        X = Dropout(self.dropout)(X)
        Y = Dense(self.n_classes, activation='softmax', name='output')(X)
        model = Model(inputs=X_input, outputs=Y)
        model.summary()
        return model

    def fit(self, train_data, steps_per_epoch=None, epochs=1, validation_data=None, validation_steps=None,
            callbacks=None, verbose=1, class_weight=None):
        for layer in self.vgg16_layers:
            layer.trainable = False
        self.model.compile(optimizer=self.optimizer_extract, loss='categorical_crossentropy', metrics=self.metrics)
        history_extract = self.model.fit(train_data, steps_per_epoch=steps_per_epoch, epochs=self.extract_epochs,
                            validation_data=validation_data, validation_steps=validation_steps, callbacks=callbacks,
                            verbose=verbose, class_weight=class_weight)
        for layer in self.vgg16_layers[self.finetune_layer:]:
            layer.trainable = True
        self.model.compile(optimizer=self.optimizer_finetune, loss='categorical_crossentropy', metrics=self.metrics)
        history_finetune = self.model.fit(train_data, steps_per_epoch=steps_per_epoch, epochs=epochs, initial_epoch=history_extract.epoch[-1],
                                      validation_data=validation_data, validation_steps=validation_steps, callbacks=callbacks,
                                      verbose=verbose, class_weight=class_weight)

    def evaluate(self, test_data, verbose=1):
        return self.model.evaluate(test_data, verbose=verbose)

    def predict(self, test_data, verbose=1):
        return self.model.predict(test_data, verbose=verbose)

    @property
    def metrics_names(self):
        return self.model.metrics_names