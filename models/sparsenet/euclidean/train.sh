#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=models/sparsenet/euclidean/ $TOOLS/caffe train \
    --solver=models/sparsenet/euclidean/solver_$1.prototxt \
    $2
    #--weights=models/sparsenet/euclidean/sparsenet_v.30.0_iter_1000000.caffemodel \
    #--gpu=1
