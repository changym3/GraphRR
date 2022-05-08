# GraphRR

This is the implementation of the paper: GraphRR: A Multiplex Graph based Reciprocal Friend Recommender System with Applications on Online Gaming Service.

Author: Yaomin Chang (changym3@mail2.sysu.edu.cn)


## Environments:
* Python
* Deep Graph Library
* PyTorch


## Overview

Here we provide the implementation of the GraphRR.

* `utils/` folder contains the the modules of data preprocessing, model architecture, and evaluation ultilizations.
* `run.py` contains the primary training process of the model.

## Reproducibility

This work is collaborated with NetEase Games and the dataset used in the paper can not be released due to the data privacy policy. 

Researchers can adapt this code into other datasets by replacing the input `dgl.Graph` with the corresponding graph.
