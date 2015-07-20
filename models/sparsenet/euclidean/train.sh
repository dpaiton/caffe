#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=models/sparsenet/euclidean/ $TOOLS/caffe train \
    --solver=models/sparsenet/euclidean/solver_$1.prototxt \
    $2
    #--gpu=1
    #--weights=models/sparsenet/euclidean/sparsenet.caffemodel
