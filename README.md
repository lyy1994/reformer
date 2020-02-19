# Reformer

PyTorch original implementation of [Neural Machine Translation with Joint Representation](https://arxiv.org/abs/2002.06546). It is modified from [fairseq-0.6.0](https://github.com/pytorch/fairseq).

## Requirements

- PyTorch version == 1.1.0
- Python version == 3.6.7

## Get started

### Installation

You need to install it first:

    git clone https://github.com/lyy1994/reformer.git
    cd reformer
    pip install -r requirements.txt
    python setup.py build develop

### Structure

Before running the training and decoding scripts, we implicitly make assumptions on the file directory structure:

    |- reformer (code)
    |- data
       |- data-bin
          |- BINARIZED_DATA_FOLDER
       |- RAW_DATA_FOLDER
          |- train (training set raw text)
          |- valid (validation set raw text)
          |- test (test set raw text)
    |- checkpoints
       |- torch-1.1.0
          |- EXPERIMENT_FOLDER
    |- toolkit
       |- multi-bleu.perl

### Usage

To train a model, run:

    cd reformer/scripts
    sh train.sh

To decode from the trained model, run:

    sh decode.sh

If you would like to customize the configuration, please modify [train.sh](https://github.com/lyy1994/reformer/blob/master/scripts/train.sh) for training and [decode.sh](https://github.com/lyy1994/reformer/blob/master/scripts/decode.sh) for decoding.

The table below summarizes the scripts for reproducing our experiments:

| Dataset | Script |
|---|---|
| IWSLT14 German-English | [iwslt-train.sh](https://github.com/lyy1994/reformer/blob/master/scripts/iwslt-train.sh) |
| NIST12 Chinese-English | [nist-train.sh](https://github.com/lyy1994/reformer/blob/master/scripts/nist-train.sh) |

## Citation

    @inproceedings{li2020aaai,
      title = {Neural Machine Translation with Joint Representation},
      author = {Yanyang Li and Qiang Wang and Tong Xiao and Tongran Liu and Jingbo Zhu},
      booktitle = {Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence},
      year = {2020},
    }
