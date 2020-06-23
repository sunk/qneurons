#!/usr/bin/env python

import numpy as np
import sys, os, warnings, re, time, argparse, uuid

sys.stdout = sys.stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter( action='ignore', category=FutureWarning )

from Qact import Counter
from gradient_noise import add_gradient_noise
import data, models

import keras
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

def contrastive_loss( y_true, y_pred ):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square( y_pred )
    margin_square = K.square( K.maximum(margin - y_pred, 0) )
    return K.mean( y_true * square_pred + (1 - y_true) * margin_square )

def siamese_accuracy( y_true, y_pred ):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def lr_schedule( epoch ):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def exp( args ):
    '''
    alg     -- elu, qelu, relu, qrelu
    lrate   -- learning rate
    dropout -- whether to dropout
    batch_size -- number of samples in a mini batch
    epochs  -- how many epochs to train
    '''

    counter = Counter()
    callbacks = [ counter ]

    if args.arch == 'mlp':
        model = models.build_mlp( args, counter )

    elif args.arch == 'cnn':
        model = models.build_cnn( args, counter )

    elif args.arch == 'resnet':
        model = models.build_resnet( args, counter )

    elif args.arch == 'siamese':
        model = models.build_siamese( args, counter )

    elif args.arch == 'ae':
        model = models.build_ae( args, counter )

    else:
        raise RuntimeError( 'unknown architecture' )

    if args.arch == 'resnet':
        if args.alg[0] == 'n':
            optimizer = add_gradient_noise( keras.optimizers.Adam )( lr=lr_schedule(0), noise_eta=args.init_v, counter=counter )
        else:
            optimizer = keras.optimizers.Adam( lr=lr_schedule(0) )

        lr_scheduler = LearningRateScheduler( lr_schedule )
        lr_reducer = ReduceLROnPlateau( factor=np.sqrt(0.1),
                                         cooldown=0,
                                         patience=5,
                                         min_lr=0.5e-6 )
        callbacks += [ lr_reducer, lr_scheduler ]

    elif args.alg[0] == 'n':
        optimizer = add_gradient_noise( keras.optimizers.SGD )( lr=args.lrate, decay=1e-6, momentum=0.5, nesterov=False, noise_eta=args.init_v, counter=counter )

    else:
        optimizer = keras.optimizers.SGD( lr=args.lrate, decay=1e-6, momentum=0.5, nesterov=False )

    if args.arch == 'siamese':
        model.compile( loss=contrastive_loss, optimizer=optimizer, metrics=[siamese_accuracy] )

        tr_pairs, tr_y = data.train_generator
        te_pairs, te_y = data.test_generator

        history = model.fit( [tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                             batch_size=args.batch_size,
                             epochs=args.epochs,
                             callbacks=callbacks,
                             validation_data=( [te_pairs[:, 0], te_pairs[:, 1]], te_y),
                             verbose=args.verbose )

        return history.history['loss'], history.history['val_siamese_accuracy']

    elif args.arch == 'ae':
        model.compile( loss=keras.losses.mse,
                       optimizer=optimizer )
        history = model.fit_generator(
                        data.train_generator,
                        steps_per_epoch=(data.x_train.shape[0]//args.batch_size+1),
                        epochs=args.epochs,
                        callbacks=callbacks,
                        validation_data=data.test_generator,
                        validation_steps=data.x_test.shape[0]//args.test_batch_size,
                        verbose=args.verbose )

        return history.history['loss'], history.history['val_loss']

    else:
        model.compile( loss=keras.losses.categorical_crossentropy,
                       optimizer=optimizer,
                       metrics=['accuracy'] )
        history = model.fit_generator(
                        data.train_generator,
                        steps_per_epoch=(data.x_train.shape[0]//args.batch_size+1),
                        epochs=args.epochs,
                        callbacks=callbacks,
                        validation_data=data.test_generator,
                        validation_steps=data.x_test.shape[0]//args.test_batch_size,
                        verbose=args.verbose )

        return history.history['loss'], history.history['val_acc']

    #score = model.evaluate( x_train, y_train, verbose=0 )
    #print( 'Train loss:', score[0] )
    #score = model.evaluate( x_test,  y_test,  verbose=0 )
    #print( 'Test accuracy:', score[1] )

def repeat_exps( args ):
    '''
    repeat training CNNs and record learning curves
    '''
    print( '-== {}-{} ==-'.format( args.dataset, args.alg ) )
    print( 'lrate={0} dropout={1} batch_size={2} epochs={3}'.format( args.lrate, args.dropout, args.batch_size, args.epochs ) )
    print( 'repeat={0}'.format( args.repeat ) )

    if args.dataset == 'mnist':
        data.load_mnist( args )
    elif args.dataset.startswith( 'cifar' ):
        data.load_cifar( args )
    else:
        print( "unknown dataset", args.dataset )
        sys.exit(0)

    curves = []
    for _ in range( args.repeat ):
        train_history, test_history = exp( args )
        print( '{:3d}'.format( _ ), end=' ' )
        print( 'train={:.4f}'.format( train_history[-1] ), end='  ' )
        print( 'test={:.4f}'.format(  test_history[-1] ) )
        curves.append( ( train_history, test_history ) )
    curves = np.array( curves )

    # save results
    filename = '{}_{}_{}'.format( args.dataset, args.arch, args.alg )
    filename += '_lr{}'.format( args.lrate )
    if args.alg.startswith( 'q' ):
        if args.const:
            filename += '_c{}'.format( args.init_v )
        else:
            filename += '_a{}'.format( args.init_v )
    elif args.alg[0] == 'n':
        filename += '_n{}'.format( args.init_v )

    if args.dropout: filename += '_dropout'

    np.savez( filename, curves=curves )
    print( 'results saved to {}'.format( filename ) )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( 'dataset',   type=str, choices=['mnist', 'cifar-10', 'cifar-100'],  help='dataset' )
    parser.add_argument( 'arch',      type=str, choices=['mlp', 'cnn', 'siamese', 'ae', 'resnet'], help='neural network architecture' )
    parser.add_argument( 'alg',       type=str, help='activation function' )

    parser.add_argument( '--lrate',   type=float, default=0.01, help='learning rate' )
    parser.add_argument( '--init_v',  type=float, default=0,    help='initial stddev for q-neuron' )
    parser.add_argument( '--const',   action='store_true',      help='use constant stddev for q-neuron' )
    parser.add_argument( '--dropout', type=int, choices=(0,1), default=0, help='enable dropout layer(s) if 1' )

    parser.add_argument( '--repeat',          type=int, default=10,  help='how many repeats for each setting' )
    parser.add_argument( '--batch_size',      type=int, default=32,  help='mini-batch size' )
    parser.add_argument( '--test_batch_size', type=int, default=256, help='test batch size (leave it to default)' )
    parser.add_argument( '--epochs',          type=int, default=200, help='#epochs' )
    parser.add_argument( '--verbose',    action='store_true',        help='verbose (only for debugging)' )
    args = parser.parse_args()

    start_t = time.time()
    repeat_exps( args )
    total_t = time.time() - start_t
    print( "finished in {0:.2f} hours".format( total_t/3600 )  )

if __name__ == '__main__':
    main()
