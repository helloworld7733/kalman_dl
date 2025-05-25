#!/bin/bash

# 创建conda环境
conda env create -f environment.yml

# 激活环境
conda activate kalman_dl

# 运行程序
python kalman_deep_learning.py 