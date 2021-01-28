# SeqGAN-Colaboratory

This repository is forked from https://github.com/MiyoshiYuto/SeqGAN. Thank you MiyoshiYuto for rewriting the SeqGan code to run in a Google Colaboratory compatible environment!

## Requirements
* **Tensorflow r2.2.0**
* Python 3.6+
* CUDA 10.1 (For GPU)
* This is compatible with the default GPU environment in Google Colaboratory :) (2020/01/28)


Note: this code is based on the [previous work by ofirnachum](https://github.com/ofirnachum/sequence_gan). Many thanks to [ofirnachum](https://github.com/ofirnachum).

## Paper
[SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](http://arxiv.org/abs/1609.05473) has been accepted at the Thirty-First AAAI Conference on Artificial Intelligence (AAAI-17).

## Motivation
This fork is intended to be used to train a SeqGan model for free in Google Colaboratory on a custom dataset (eg. movie reviews). I hope that this makes the original art more accessible.

## File descriptions
In addition to the original code, I have added the following files:
* prepare_nltk_data.ipynb - Download the nltk movie review dataset and save for cunsumption by `sequence_gan.py` (for SeqGan model training).
* prepare_custom_data.ipynb, prepare_custom_data.py - Code to prepare a custom text dataset for cunsumption by `sequence_gan.py`.
* seq_gan_train.ipynb - An example Colab notebook for SeqGan training. Use Colab's free GPU environment!
* seq_gan_run.ipynb - An example Colab notebook for generating new sequences using a saved SeqGan.

I have also removed the originat TARGET_LSTM oracle model and .pkl since this code is intended to be used on real text data. An example of movie review text data is svaed in `./data/`. You can also create your own data using either `prepare_nltk_data.ipynb`, `prepare_custom_data.ipynb` or `prepare_custom_data.py`.

## How to train a model
Simply upload the notebook `seq_gan_train.ipynb` to Colaboratory and run it!

