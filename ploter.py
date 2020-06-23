#!/usr/bin/env python

'''
generate figure based on the .npz files
produced by runqact.py
'''

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['legend.fontsize'] = 10
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import os, sys, re
from collections import defaultdict

def mae( interval, window_size ):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve( interval, window, 'valid' )

def visualize( data, arch, names, curve_means, curve_stds ):

    fig = plt.figure( figsize=(5,4), dpi=300,  constrained_layout=True )
    ax1 = fig.add_subplot(111)
    ax2  = ax1.twinx()

    colors = {     names[0] : 'crimson',
               'n'+names[0] : 'forestgreen',
               'q'+names[0] : 'royalblue',    # blue is our method
        names[0]+'+Dropout' : 'violet',
                      'ReLU': 'tab:orange',
                    'swish' : 'tab:cyan' }

    # small hack to determine if the task is supervised
    initial_test = np.nanmean( curve_means[0,1,:5] )
    final_test = np.nanmean( curve_means[0,1,-5:] )
    supervised = ( initial_test < final_test )

    nepochs = curve_means.shape[-1]
    print( '#epochs: ', nepochs )
    print( 'supervised:', supervised )

    # smooth the curves based on moving average
    smoothed = []
    for _mean in curve_means:
        smoothed.append( ( mae(_mean[0],5), mae(_mean[1],5) ) )
    curve_means = np.array( smoothed )

    for _name, _mean, _std in zip( names, curve_means, curve_stds ):
        print( 'processing {0}'.format( _name ) )

        if _name.startswith( 'q' ):
            lw = 2
            alpha = 1
        else:
            lw = 1.4
            alpha = .8

        _c = colors[_name]
        if supervised:
            ax1.plot( _mean[0], ls='--', lw=lw, alpha=alpha, color=_c )
            ax2.plot( _mean[1], ls='-',  lw=lw, alpha=alpha, color=_c, label='{0} ({1:.2f})'.format( _name, 100*(1-_mean[1,-1]) ) )
        else:
            ax1.plot( _mean[0], ls='--',   lw=lw, alpha=alpha, color=_c )
            ax2.plot( 1-_mean[1], ls='-',  lw=lw, alpha=alpha, color=_c, label='{0} ({1:.4f})'.format( _name, _mean[1,-1] ) )

    if supervised:
        ax1.set_xlabel( 'epoch' )
        ax1.set_ylabel( 'training loss' )
        ax1.set_xlim( 0, nepochs-5 )
        ax1.set_ylim( curve_means[:,0,20:].min()-0.01,  0.7 ) # curve_means[:,0,20:].max() )
        #ax1.set_ylim( curve_means[:,0,10:].min(), 0.03 )

        ax2.set_ylabel( 'testing error (percentage)' )
        ax2.set_ylim( 0.8, curve_means[:,1,20:].max()+0.002 )
#curve_means[:,1,20:].min(),
        #ax2.set_ylim( 0.96, 0.999 ) #curve_means[:,1,10:].max() )

        labels = ax2.get_yticks()
        ax2.set_yticklabels( [ '{:g}'.format(100-100*float(l)) for l in labels ] )
        ax2.legend( loc=5, fancybox=True, framealpha=0.8 )

    else:
        ax1.set_xlabel( 'epoch' )
        ax1.set_ylabel( 'training loss' )
        ax1.set_xlim( 0, nepochs-1 )
        ax1.set_ylim( curve_means[:,0,10:].min(), 0.002 ) # curve_means[:,0,10:].max() )

        ax2.set_ylabel( 'testing loss' )
        ax2.set_ylim( 1-0.002, 1-curve_means[:,1,10:].min() )#, 0.001 ) #curve_means[:,1,10:].max() )

        labels = ax2.get_yticks()
        ax2.set_yticklabels( [ '{:g}'.format(1-float(l)) for l in labels ] )
        ax2.legend( loc=5, fancybox=True, framealpha=0.8 )

    title = '{} {}'.format( data, arch )
    ax1.set_title( title )
    title = title.replace( ' ', '_' )
    ofilename = '{}.pdf'.format( title )
    print( "result saved to {}".format( ofilename ) )

    plt.savefig( ofilename, transparent=True )

def parse_filename( filename ):
    '''
    from raw filename to data, arch, method
    '''
    basename = os.path.splitext( os.path.basename( filename ) )[0]
    fields = basename.split( '_' )

    data, arch, method = fields[:3]
    if 'dropout' in filename.lower():
        method += '+Dropout'

    # unify string names

    if data.lower() == 'mnist':
        data = 'MNIST'
    elif data.lower() in ( 'cifar10', 'cifar-10' ):
        data = 'CIFAR-10'
    elif data.lower() in ( 'cifar100', 'cifar-100' ):
        data = 'CIFAR-100'

    if arch.lower() == 'mlp':
        arch = 'MLP'
    elif arch.lower() == 'cnn':
        arch = 'CNN'
    elif arch.lower().startswith( 'resnet' ):
        arch = 'ResNet-56'
    elif arch.lower() in ( 'ae', 'autoencoder' ):
        arch = 'Autoencoder'
    elif arch.lower() == 'siamese':
        arch = 'Siamese'

    # replace the first match
    NAME_DICT = [
        ('relu', 'ReLU'),
        ('elu', 'ELU'),
    ]
    for old, new in NAME_DICT:
        if old in method:
            method = method.replace( old, new )
            break

    return data, arch, method

def get_files():
    if len( sys.argv ) > 1:
        root = sys.argv[1]
    else:
        root = "."

    if len( sys.argv ) > 2:
        keys = sys.argv[2:]
    else:
        keys = ['']

    filenames = []
    for dirs, subdirs, files in os.walk( root ):
        for _filename in files:
            fullname = os.path.join( dirs, _filename )
            if os.access( fullname, os.R_OK ) and _filename.endswith( 'npz' ):
                haskey = True
                for key in keys:
                    if not key in _filename: haskey = False
                if not haskey: continue
                filenames.append( fullname )

    return filenames

def main():
    '''
    read learning curves from input files
    and visualize these curves into pdf files
    '''

    results = defaultdict( list )
    for _file in get_files():
        data, arch, method = parse_filename( _file )

        _curves = np.load( _file )['curves']
        n_curves = _curves.shape[0]
        _curves = np.array( [ arr for arr in _curves if arr[0,-1] < arr[0,0] ] )
        if _curves.shape[0] < n_curves:
            print( f'throwing away {n_curves-_curves.shape[0]} failed runs from {_file}' )
        if _curves.size == 0: continue

        _mean = _curves.mean(0)
        _std  = _curves.std(0)

        results[method].append( (_file, _mean, _std) )

    for _key in results:
        # sort by training loss
        # results[_key].sort( key=lambda r: r[1][0,-1] )

        if arch == 'Autoencoder':
            # sort by testing loss
            results[_key].sort( key=lambda r: r[1][1,-5:].mean() )
        else:
            # sort by testing accuracy of the last five epochs
            results[_key].sort( key=lambda r: r[1][1,-5:].mean(), reverse=True )

        for _file, _mean, _std in results[_key]:
            print( "{:20s} {:50s} train:{:.4f} test:{:.4f}".format( _key, _file, _mean[0,-1], _mean[1,-1] ) )

        results[_key] = results[_key][0]

    names        = []
    curve_means  = []
    curve_stds   = []

    def __curve( __method ):
        _file, _mean, _std = results.pop( __method )

        names.append( __method )
        curve_means.append( _mean )
        curve_stds.append( _std )

    act = min( results.keys(), key=lambda x:len(x) )
    __curve( act )
    if 'n'+act in results: __curve( 'n'+act )
    if 'q'+act in results: __curve( 'q'+act )
    if act+'+Dropout' in results: __curve( act+'+Dropout' )

    rest = list( results.keys() )
    for __method in rest: __curve( __method )

    visualize( data, arch, names, np.array( curve_means ), np.array( curve_stds ) )

if __name__ == '__main__':
    main()
