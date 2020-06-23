import keras
from keras import backend as K
from keras.datasets.cifar import load_batch
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os

def __load_cifar10():
    """Loads CIFAR10 dataset. """

    path = os.path.expanduser( "~/.keras/datasets/cifar-10-batches-py" )

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test  = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)

def __load_cifar100():
    """Loads CIFAR100 dataset.

    # Arguments
        label_mode: one of "fine", "coarse".

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    # Raises
        ValueError: in case of invalid `label_mode`.
    """
    label_mode ='fine'
    path = os.path.expanduser( "~/.keras/datasets/cifar-100-python" )

    fpath = os.path.join(path, 'train')
    x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(path, 'test')
    x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)

def __load_mnist():
    """Loads the MNIST dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    # download the file from
    # https://s3.amazonaws.com/img-datasets/mnist.npz

    path = os.path.expanduser( "~/.keras/datasets/mnist.npz" )
    f = np.load( path )

    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()

    return (x_train, y_train), (x_test, y_test)

def load_cifar( args ):
    '''
    load cifar10 or cifar100
    '''
    global x_train, y_train, x_test, y_test, train_generator, test_generator

    if args.dataset == 'cifar-10':
        (x_train, y_train), (x_test, y_test) = __load_cifar10()
        num_classes = 10
    else:
        (x_train, y_train), (x_test, y_test) = __load_cifar100()
        num_classes = 100

    x_train = x_train.astype( 'float32' )
    x_test  = x_test.astype( 'float32' )
    x_train /= 255
    x_test  /= 255

    print( x_train.shape, x_test.shape )

    if args.arch == 'siamese':
        digit_indices = [ np.where(y_train == i)[0] for i in range(num_classes) ]
        train_generator = create_pairs( x_train, digit_indices )

        digit_indices = [ np.where(y_test == i)[0] for i in range(num_classes) ]
        test_generator = create_pairs(x_test, digit_indices)

    elif args.arch == 'ae':
        train_generator = ImageDataGenerator().flow( x_train, x_train, batch_size=args.batch_size )
        test_generator  = ImageDataGenerator().flow( x_test,  x_test,  batch_size=args.test_batch_size )

    elif args.arch == 'resnet':
        # use the same setting as the official example
        x_train_mean = np.mean( x_train, axis=0 )
        x_train -= x_train_mean
        x_test  -= x_train_mean

        y_train = keras.utils.to_categorical( y_train, num_classes )
        y_test  = keras.utils.to_categorical( y_test,  num_classes )

        train_generator = ImageDataGenerator(
                            # set input mean to 0 over the dataset
                            featurewise_center=False,
                            # set each sample mean to 0
                            samplewise_center=False,
                            # divide inputs by std of dataset
                            featurewise_std_normalization=False,
                            # divide each input by its std
                            samplewise_std_normalization=False,
                            # apply ZCA whitening
                            zca_whitening=False,
                            # epsilon for ZCA whitening
                            zca_epsilon=1e-06,
                            # randomly rotate images in the range (deg 0 to 180)
                            rotation_range=0,
                            # randomly shift images horizontally
                            width_shift_range=0.1,
                            # randomly shift images vertically
                            height_shift_range=0.1,
                            # set range for random shear
                            shear_range=0.,
                            # set range for random zoom
                            zoom_range=0.,
                            # set range for random channel shifts
                            channel_shift_range=0.,
                            # set mode for filling points outside the input boundaries
                            fill_mode='nearest',
                            # value used for fill_mode = "constant"
                            cval=0.,
                            # randomly flip images
                            horizontal_flip=True,
                            # randomly flip images
                            vertical_flip=False,
                            # set rescaling factor (applied before any other transformation)
                            rescale=None,
                            # set function that will be applied on each input
                            preprocessing_function=None,
                            # image data format, either "channels_first" or "channels_last"
                            data_format=None,
                            # fraction of images reserved for validation (strictly between 0 and 1)
                            validation_split=0.0).flow( x_train, y_train, batch_size=args.batch_size )
        test_generator  = ImageDataGenerator().flow( x_test, y_test, batch_size=args.test_batch_size )

    else:
        y_train = keras.utils.to_categorical( y_train, num_classes )
        y_test  = keras.utils.to_categorical( y_test,  num_classes )

        train_generator = ImageDataGenerator(
                          featurewise_center=False,
                          samplewise_center=False,
                          featurewise_std_normalization=False,
                          samplewise_std_normalization=False,
                          zca_whitening=False,
                          rotation_range=5,
                          width_shift_range=0.1,
                          height_shift_range=0.1,
                          horizontal_flip=True,
                          vertical_flip=True ).flow( x_train, y_train, batch_size=args.batch_size )
        test_generator  = ImageDataGenerator().flow( x_test, y_test, batch_size=args.test_batch_size )

def load_mnist( args, IMG_ROWS=28, IMG_COLS=28, NUM_CLASSES=10 ):
    '''
    load mnist dataset
    '''
    global x_train, y_train, x_test, y_test, train_generator, test_generator

    (x_train, y_train), (x_test, y_test) = __load_mnist()
    print( x_train.shape, x_test.shape )

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, IMG_ROWS, IMG_COLS )
        x_test  = x_test.reshape( x_test.shape[0], 1, IMG_ROWS, IMG_COLS  )

    else:
        x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1 )
        x_test  = x_test.reshape( x_test.shape[0], IMG_ROWS, IMG_COLS, 1  )

    x_train = x_train.astype( 'float32' )
    x_test  = x_test.astype( 'float32' )
    x_train /= 255
    x_test  /= 255

    if args.arch == 'siamese':
        digit_indices = [ np.where(y_train == i)[0] for i in range(NUM_CLASSES) ]
        train_generator = create_pairs( x_train, digit_indices )

        digit_indices = [ np.where(y_test == i)[0] for i in range(NUM_CLASSES) ]
        test_generator = create_pairs(x_test, digit_indices)

    elif args.arch == 'ae':
        train_generator = ImageDataGenerator().flow( x_train, x_train, batch_size=args.batch_size )
        test_generator  = ImageDataGenerator().flow( x_test,  x_test,  batch_size=args.test_batch_size )

    else:

        y_train = keras.utils.to_categorical( y_train, NUM_CLASSES )
        y_test  = keras.utils.to_categorical( y_test,  NUM_CLASSES )

        train_generator = ImageDataGenerator(
                          rotation_range=8,
                          width_shift_range=0.08,
                          shear_range=0.3,
                          height_shift_range=0.08,
                          zoom_range=0.08 ).flow( x_train, y_train, batch_size=args.batch_size )
        test_generator  = ImageDataGenerator().flow( x_test, y_test, batch_size=args.test_batch_size )

def create_pairs( x, digit_indices ):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min( [ len(_indices) for _indices in digit_indices ] ) - 1

    num_classes = len( digit_indices )
    chaos = np.random.RandomState( 2020 )

    for d in range( num_classes ):
        for i in range( n ):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [ [x[z1], x[z2]] ]

            inc = chaos.randint( 1, num_classes )
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]

            labels += [1, 0]
    return np.array(pairs), np.array(labels)
