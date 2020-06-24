# $q$-Neurons: Neuron Activations based on Stochastic Jackson's Derivative Operators

An application of quantum calculus [1] in deep learning and an extension of our technical report [2]. A $q$-neuron is a stochastic neuron with its activation function relying on Jackson's discrete $q$-derivative for a stochastic parameter $q$.

## Requirements

- Python >= 3.6.x
- keras-2.2.x
- tensorflow-1.1x

## Run the code

```bash
python runqact.py <mnist|cifar-10|cifar-100> <mlp|cnn|siamese|ae|resnet> <elu|qelu|nelu>
```
where ```elu``` means Exponential Linear Units, ```qelu``` means its corresponding q-activation, and ```nelu``` means gradient noise injection [2]. See
```bash
python runqact.py --help
```
for more configuration options.

To repeat our reported results, see the scripts in the ```hpc``` directory.

## References

[1] Victor Kac and Pokman Cheung, [Quantum Calculus](https://www.springer.com/gp/book/9780387953410), Springer-Verlag New York, 2001.

[2] Frank Nielsen, Ke Sun, [$q$-Neurons: Neuron Activations based on Stochastic Jackson's Derivative Operators](https://arxiv.org/abs/1806.00149), arXiv:1806.00149, 2018.

[3] Neelakantan et al., [Adding gradient noise improves learning for very deep networks](https://arxiv.org/abs/1511.06807), in International Conference on Learning Representations workshop, 2016.

## Cite

@article{qneurons,
  author={Frank Nielsen and Ke Sun},
  title={$q$-{N}eurons: {N}euron Activations based on Stochastic {J}ackson's Derivative Operators},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  pages={(to appear)},
  year={2020}
}
