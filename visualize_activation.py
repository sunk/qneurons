#!/usr/bin/env python

import numpy as np

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def relu( x ):
    return x * (x>=0)

def drelu( x ):
    return (x>=0)

def sin( x ):
    return np.sin( x )

def softplus( x ):
    return np.log( 1 + np.exp(x) )

def elu( x, alpha=0.5 ):
    return x * (x>=0) + alpha * ( np.exp(x)-1 ) * (x<0)

def delu( x, alpha=0.5 ):
    return (x>=0) + alpha * np.exp(x) * (x<0)

def sigmoid( x ):
    return np.exp( -np.logaddexp(0, -x) )

def dsigmoid( x ):
    return sigmoid(x) * (1-sigmoid(x))

def requ( x ):
    return x*x*(x>=0)

def drequ( x ):
    return 2*x*(x>=0)

def tanh( x ):
    return np.tanh( x )

def swish(x):
    return x * sigmoid( x )

def visualize( ax, act, mx, my, stddev, ylabel, title ):
    N = 6000
    RES = 600

    density = np.zeros( [RES, RES] )
    for i, x in enumerate( np.linspace( -mx, mx, RES ) ):
        epsilon = np.random.randn( N )
        q = 1 + (2 * (epsilon>=0) - 1 ) * (stddev * np.abs(epsilon) + 0.001 )
        y = ( act( q*x ) - act(x) ) / (q-1)

        y_cor = ( 0.5*(y+my)/my * RES ).astype(np.int)
        y_cor = np.extract( y_cor>=0, y_cor )
        y_cor = np.extract( y_cor<RES, y_cor )

        np.add.at( density[i], y_cor, 1 )

    from matplotlib.colors import ListedColormap
    cmap = cm.Oranges
    my_cmap = cmap( np.arange(cmap.N) )
    my_cmap[:,-1] = np.ones( cmap.N )
    my_cmap[0,-1] = 0
    my_cmap = ListedColormap( my_cmap )

    ax.imshow( density.T, origin='lower', cmap=my_cmap, vmin=0, vmax=2*N/RES, interpolation='hamming' )

    x = np.linspace( -mx, mx, 10000 )
    y = act(x)
    x_cor = 0.5 * ( x + mx ) / mx * RES
    y_cor = 0.5 * ( y + my ) / my * RES
    ax.plot( x_cor, y_cor, '-', color='b', linewidth=4, alpha=.2 )

    ax.axhline( y=RES/2, lw=0.5, color=cmap(1.0), alpha=.1 )
    ax.axvline( x=RES/2, lw=0.5, color=cmap(1.0), alpha=.1 )

    ax.set_xlim( [0, RES] )
    ax.set_ylim( [0, RES] )
    ax.set_xticks( [0,(RES-1)/2,RES-1] )
    ax.set_xticklabels( [-mx, 'x', mx] )
    ax.set_yticks( [0,(RES-1)/2,RES-1] )
    ax.set_yticklabels( [-my, ylabel, my], rotation=90 )

    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_verticalalignment( 'center' )

    ax.set_title( title )

def main():
    fig = plt.figure( figsize=(6,9), dpi=300 )

    ax = fig.add_subplot( 321 )
    visualize( ax, tanh, 3, 1.1, 0.1, r'$\lambda=0.1$', '$q$tanh' )

    ax = fig.add_subplot( 322 )
    visualize( ax, sin, 5, 3.5,   0.1, r'$\lambda=0.1x$', '$q$sin' )

    ax = fig.add_subplot( 323 )
    visualize( ax, elu,  5, 1,  0.1, r'$\lambda=0.1$', '$q$ELU' )

    ax = fig.add_subplot( 324 )
    visualize( ax, requ, 3, 4,  0.1, r'$\lambda=0.1$', '$q$ReQU' )

    ax = fig.add_subplot( 325 )
    visualize( ax, swish, 5, 1, 0.1, r'$\lambda=0.1$', '$q$Swish' )

    ax = fig.add_subplot( 326 )
    visualize( ax, relu, 2, 2,  0.3, r'$\lambda=0.3$', '$q$ReLU' )

    plt.savefig( 'qneurons.pdf', bbox_inches='tight', pad_inches=0, transparent=True )

if __name__ == '__main__':
    main()
