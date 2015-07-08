#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=models/sparsenet_mnist/ $TOOLS/caffe train \
  --solver=models/sparsenet_mnist/solver.prototxt
    #--gpu=$1
    #--weights=models/sparsenet_mnist/sparsenet.caffemodel \
