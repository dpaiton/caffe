#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=models/sparsenet_logistic/ $TOOLS/caffe train \
  --solver=models/sparsenet_logistic/solver_$1.prototxt
    #--gpu=$2
    #--weights=models/sparsenet_logistic/sparsenet.caffemodel \
