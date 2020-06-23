import data
from Qact import Qact, Counter

import keras
from keras import backend as K

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Lambda, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, AveragePooling2D, Reshape, Conv2DTranspose, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.generic_utils import get_custom_objects

def swish( x ):
    return K.sigmoid(x) * x
get_custom_objects().update( {'swish': Activation(swish)} )

def activate( args, counter ):
    '''
    wrapper of activation layer
    '''
    _ACT_ = [ 'relu', 'elu', 'selu', 'tanh', 'sigmoid', 'softplus' ]
    anneal_mode = 0 if args.const else 2

    if args.alg.startswith( 'q' ) and args.alg[1:] in _ACT_:
        return Qact( args.alg, counter, args.init_v, anneal_mode=anneal_mode )

    elif args.alg.startswith( 'n' ) and args.alg[1:] in _ACT_:
        return Activation( args.alg[1:] )

    elif args.alg.lower() == 'swish':
        return Activation( 'swish' )

    elif args.alg in _ACT_:
        # otherwise alg is keras-predefined activation
        return Activation( args.alg )

    else:
        raise RuntimeError( 'unknown algorithm: {}'.format( args.alg ) )

def build_mlp( args, counter ):
    '''
    a simple MLP with two hidden layers
    '''
    model = Sequential()

    model.add( Flatten( input_shape=data.x_train.shape[1:] ) )

    model.add( Dense( 256 ) )
    model.add( activate( args, counter ) )
    if args.dropout: model.add( Dropout(0.25) )

    model.add( Dense( 256 ) )
    model.add( activate( args, counter ) )
    if args.dropout: model.add( Dropout(0.125) )

    model.add( Dense(data.y_train.shape[1], activation='softmax') )
    return model

def build_siamese( args, counter ):
    '''
    a simple MLP with two hidden layers
    '''
    if K.image_data_format() == 'channels_first':
        norm_axis = 1
    else:
        norm_axis = -1

    def create_base_network( input_shape ):
        '''Base network to be shared (eq. to feature extraction).'''

        model = Sequential()
        model.add( Conv2D( 32, (3, 3), input_shape=input_shape ) )
        model.add( MaxPooling2D(pool_size=(2,2)) )
        model.add( activate( args, counter ) )
        model.add( BatchNormalization( axis=norm_axis ) )

        model.add( Conv2D(64, (3, 3) ) )
        model.add( MaxPooling2D(pool_size=(2,2)) )
        model.add( activate( args, counter ) )
        model.add( BatchNormalization( axis=norm_axis ) )

        model.add( Conv2D(64, (3, 3) ) )
        model.add( MaxPooling2D(pool_size=(2,2)) )
        model.add( activate( args, counter ) )

        model.add( Flatten() )
        model.add( BatchNormalization() )
        if args.dropout: model.add( Dropout(0.125) )

        model.add( Dense(64) )
        model.add( activate( args, counter ) )
        model.add( BatchNormalization() )

        model.add( Dense(64) )
        model.summary()
        return model

    def euclidean_distance( vects ):
        x, y = vects
        sum_square = K.sum( K.square(x - y), axis=1, keepdims=True )
        return K.sqrt( K.maximum(sum_square, K.epsilon()) )

    def eucl_dist_output_shape( shapes ):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    input_shape = data.x_train.shape[1:]
    base_network = create_base_network( input_shape )
    input_a = Input( shape=input_shape )
    input_b = Input( shape=input_shape )
    processed_a = base_network( input_a )
    processed_b = base_network( input_b )

    distance = Lambda( euclidean_distance,
                       output_shape=eucl_dist_output_shape)( [processed_a, processed_b] )

    return Model([input_a, input_b], distance)

def build_ae( args, counter ):
    '''A convolutional autoencoder'''
    if K.image_data_format() == 'channels_first':
        norm_axis = 1
    else:
        norm_axis = -1

    model = Sequential()

    shape = data.x_train.shape[1:]

    model.add( Conv2D(32, (3, 3), strides=2, input_shape=shape, padding='same') )
    model.add( activate( args, counter ) )

    model.add( Conv2D(32, (3, 3), strides=2, padding='same') )
    model.add( activate( args, counter ) )

    if args.dropout: model.add( Dropout(0.25) )

    model.add( Conv2DTranspose(32, (3, 3), strides=2, padding='same') )
    model.add( activate( args, counter ) )

    model.add( Conv2DTranspose(32, (3, 3), strides=2, padding='same') )
    model.add( activate( args, counter ) )

    model.add( Conv2D(shape[norm_axis], (3, 3), padding='same') )

    return model

def build_cnn( args, counter ):
    if K.image_data_format() == 'channels_first':
        norm_axis = 1
    else:
        norm_axis = -1

    # the following CNN architecture can get >=99.5% testing accuracy (MNIST)
    model = Sequential()
    model.add( Conv2D(32, (3, 3), input_shape=data.x_train.shape[1:]) )
    model.add( activate( args, counter ) )
    #model.add( BatchNormalization( axis=norm_axis ) )

    model.add( Conv2D(32, (3, 3)) )
    model.add( activate( args, counter ) )

    model.add( MaxPooling2D(pool_size=(2,2)) )
    #model.add( BatchNormalization( axis=norm_axis ) )

    model.add( Conv2D(64, (3, 3)) )
    model.add( activate( args, counter ) )
    #model.add( BatchNormalization( axis=norm_axis ) )

    model.add( Conv2D(64, (3, 3)) )
    model.add( activate( args, counter ) )

    model.add( MaxPooling2D(pool_size=(2,2)) )

    model.add( Flatten() )
    #model.add( BatchNormalization() )

    model.add( Dense(512) )
    model.add( activate( args, counter ) )
    #model.add( BatchNormalization() )

    # optional dropout (which improves testing accuracy)
    if args.dropout: model.add( Dropout(0.2) )

    # the output layer
    model.add( Dense(data.y_train.shape[1], activation='softmax') )
    model.summary()
    return model

def resnet_layer( inputs,
                  args,
                  counter,
                  num_filters=16,
                  kernel_size=3,
                  strides=1,
                  batch_normalization=True,
                  conv_first=True,
                  activation=True ):
    """2D Convolution-Batch Normalization-Activation stack builder"""

    conv = Conv2D( num_filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4) )

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation:
            x = activate( args, counter )(x)

    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation:
            x = activate( args, counter )(x)
        x = conv(x)

    return x

def resnet_v2( input_shape, depth, num_classes, args, counter ):
    """ResNet Version 2 Model builder [b] """

    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input( shape=input_shape )
    x = resnet_layer(inputs, args, counter,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = True
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = False
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer( x, args, counter,
                              num_filters=num_filters_in,
                              kernel_size=1,
                              strides=strides,
                              activation=activation,
                              batch_normalization=batch_normalization,
                              conv_first=False )
            y = resnet_layer( y, args, counter,
                              num_filters=num_filters_in,
                              conv_first=False )
            y = resnet_layer( y, args, counter,
                              num_filters=num_filters_out,
                              kernel_size=1,
                              conv_first=False )
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer( x, args, counter,
                                  num_filters=num_filters_out,
                                  kernel_size=1,
                                  strides=strides,
                                  activation=False,
                                  batch_normalization=False )
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = activate( args, counter )(x)
    x = AveragePooling2D( pool_size=(int(x.shape[1]), int(x.shape[2])) )(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_resnet( args, counter, depth=56 ):
    return resnet_v2( data.x_train.shape[1:], depth, data.y_train.shape[1], args, counter )
