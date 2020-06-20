# Using PILCO for continuum robots control
[![Build Status](https://travis-ci.org/nrontsis/PILCO.svg?branch=master)](https://travis-ci.org/nrontsis/PILCO)
[![codecov](https://codecov.io/gh/nrontsis/PILCO/branch/master/graph/badge.svg)](https://codecov.io/gh/nrontsis/PILCO)

**This is part of the code of [Efficient reinforcement learning control for continuum robots based on Inexplicit Prior Knowledge](https://arxiv.org/abs/2002.11573).
The main repository of this paper is [Skylark0924/TendonTrack](https://github.com/Skylark0924/TendonTrack).**

An implementation of model-based reinforcement learning control for continuum robots using modern \& clean version of the [PILCO](https://ieeexplore.ieee.org/abstract/document/6654139/) Algorithm in `TensorFlow v2`.

Unlike PILCO's [original implementation](http://mlg.eng.cam.ac.uk/pilco/) which was written as a self-contained package of `MATLAB`, this repository aims to provide a clean implementation by heavy use of modern machine learning libraries.

In particular, we use `TensorFlow v2` to avoid the need for hardcoded gradients and scale to GPU architectures. Moreover, we use [`GPflow v2`](https://github.com/GPflow/GPflow) for Gaussian Process Regression.

The core functionality is tested against the original `MATLAB` implementation.

## Example of usage
Before using `PILCO` you have to install it by running:
```
https://github.com/Skylark0924/TendonTrack_PILCO.git && cd TendonTrack_PILCO
python setup.py develop
```
It is recommended to install everything in a fresh conda environment with `python>=3.7`

The examples included in this repo use [`OpenAI gym 0.15.3`](https://github.com/openai/gym#installation) and [`mujoco-py 2.0.2.7`](https://github.com/openai/mujoco-py#install-mujoco). Theses dependecies should be installed manually. Then, you can run one of the examples as follows
```
python examples/gym_tracking_tendon.py
```


## References

See the following publications for a description of the algorithm: [1](https://ieeexplore.ieee.org/abstract/document/6654139/), [2](http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf), 
[3](https://pdfs.semanticscholar.org/c9f2/1b84149991f4d547b3f0f625f710750ad8d9.pdf)