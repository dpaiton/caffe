#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=models/sparsenet/logistic/ $TOOLS/caffe train \
    --solver=models/sparsenet/logistic/solver_$1.prototxt \
    $2
    #--gpu=1
    #--weights=models/sparsenet/logistic/sparsenet.caffemodel 
