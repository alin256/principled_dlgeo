
![Python 3.6](https://img.shields.io/badge/python-3.8-green.svg)
![Tensorflow 2.9](https://img.shields.io/badge/tensorflow-2.9-orange)

# A principled deep adversarial learning approach for geological facies generation

ArXiv : https://arxiv.org/abs/2305.13318

The simulation of geological facies in an unobservable volume is essential in various geoscience applications. Given the complexity of the problem, deep generative learning is a promising approach to overcome the limitations of traditional geostatistical simulation models, in particular their lack of physical realism. This research aims to investigate the application of generative adversarial networks and deep variational inference for conditionally simulating meandering channelized reservoir in underground volumes. In this paper, we review the generative deep learning approaches, in particular the adversarial ones and the stabilization techniques that aim to facilitate their training. We also study the problem of conditioning deep learning models to observations through a variational Bayes approach, comparing a conditional neural network model to a Gaussian mixture model.
The proposed approach is tested on 2D and 3D simulations generated by the stochastic process-based model Flumy. Morphological metrics are utilized to compare our proposed method with earlier iterations of generative adversarial networks. The results indicate that by utilizing recent stabilization techniques, generative adversarial networks can efficiently sample from target data distributions.

![Image of 3D Flumy blocs generated (unable to load)](./images/bloc_3d.jpg)

**Hardware requirements:** A GPU CUDA compatible installed is highly recommended. 16GB of System RAM, as well as GPU RAM, is recommended for 3D codes.

**Software requirements:** Listed in requirements.txt

**The following codes do not belong to us:** spectral_normalization_layers.py: SpectralNormalization in Keras (Source: https://github.com/IShengFang/SpectralNormalizationKeras)

**Contact**: ferdinand.bhavsar@minesparis.psl.eu
