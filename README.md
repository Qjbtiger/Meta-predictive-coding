# Meta predictive learning (MPL) model

Code for [Meta predictive learning model of natural languages](https://arxiv.org/abs/2309.04106.pdf) (Arxiv: 2309.04106). The back-propagation (BP) algorithm is an efficient algorithm in training neural networks, but it is not the way our brain work. To bridge the connection between artificial nerual networks and the human brain, we demonstrate the meta predictive learning model, which considers the training method called predictive coding and the weights in neural networks satisfying spike-and-slab distribution. We trained the models on MNIST, a toy model, and a real corpus (Penn Treebank) and achieved the same level as the BP algorithm.

## Requirements

- Python 3.10
- Numpy 1.22.4
- Pytorch 1.13 with Cuda 11.6 (Train on PTB only)
- torchtext 0.14 (Train on PTB only)
- torchdata 0.5 (Train on PTB only)

## Acknowledgment

- [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/)
- [Penn Treebank Corpus](https://catalog.ldc.upenn.edu/docs/LDC95T7/cl93.html)

## Citation

This code is the product of work carried out by the group of [PMI lab, Sun Yat-sen University](https://www.labxing.com/hphuang2018). If the code helps, consider giving us a shout-out in your publications.
