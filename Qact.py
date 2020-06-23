import keras
import keras.backend as K
import numpy as np

class Qact( keras.engine.Layer ):
    """
    Quantum activation
    """

    def __init__( self, act, counter, init_stddev=1., anneal_mode=2, **kwargs ):
        '''
                act: keras activation layer
            counter: counter object (callback)
        init_stddev: initial stddev
        anneal_mode: 0 constant       (constant noise scale already works well)
                     1 init_stddev / (1+epoch)
                     2 init_stddev / (1+0.5*epoch)
        '''
        super( Qact, self ).__init__( **kwargs )

        self.supports_masking = True
        self.eps              = 1e-3

        if act == 'qselu':
            self.act  = keras.activations.selu

        elif act == 'qelu':
            self.act  = keras.activations.elu

        elif act == 'qsigmoid':
            self.act  = keras.activations.sigmoid

        elif act == 'qtanh':
            self.act = keras.activations.tanh

        else:
            raise NotImplementedError()

        self.counter          = counter
        self.init_stddev      = init_stddev
        self.anneal_mode      = anneal_mode

    def build( self, input_shape ):
        super( Qact, self ).build( input_shape )

    def call( self, inputs ):
        if self.anneal_mode == 0:
            _stddev = self.init_stddev

        elif self.anneal_mode == 1:
            _stddev = self.init_stddev / (1+self.counter.nepoch)

        elif self.anneal_mode == 2:
            _stddev = self.init_stddev / (1+0.5*self.counter.nepoch)

        else:
            raise RuntimeError( "unknown annealing mode {}".format( self.anneal_mode ) )

        # ensure that Q is "normally" distributed and non-zero
        Q = K.random_normal( shape=K.shape(inputs) )
        Q = _stddev * Q + K.switch( Q>=0.0, self.eps*K.ones_like(Q), -self.eps*K.ones_like(Q) )

        # Q = _stddev * K.abs( Q ) + self.eps # alternative

        # this one is unbiased
        #return -( self.act( inputs * (1+Q) ) - (1+Q) * self.act( inputs ) ) / Q + self.dactx(inputs)

        # this simple formulation is faster and performs better
        return ( self.act( inputs * (1+Q) ) - self.act( inputs ) ) / Q

    def get_config( self ):
        config = { 'init_stddev': self.init_stddev,
                   'anneal_mode': self.anneal_mode, }
        base_config = super( Qact, self ).get_config()
        return dict( list(base_config.items()) + list(config.items()) )

    def compute_output_shape( self, input_shape ):
        return input_shape

class Counter( keras.callbacks.Callback ):
    def __init__( self, **kwargs ):
        super( Counter, self ).__init__( **kwargs )
        self.nepoch  = K.variable( np.array(0), dtype='float32', name='nepoch' )
        self.nbatch  = K.variable( np.array(0), dtype='float32', name='nbatch' )

    def on_train_begin( self, logs={} ):
        K.eval( self.nepoch.assign(0) )
        K.eval( self.nbatch.assign(0) )

        self.inc_epoch = self.nepoch.assign( self.nepoch + 1 )
        self.inc_batch = self.nbatch.assign( self.nbatch + 1 )

    def on_train_end( self, logs={} ):
        return

    def on_epoch_begin( self, epoch, logs={} ):
        return

    def on_epoch_end( self, epoch, logs={} ):
        K.eval( self.inc_epoch )

    def on_batch_begin( self, batch, logs={} ):
        return

    def on_batch_end( self, batch, logs={} ):
        K.eval( self.inc_batch )

