#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=models/sparsenet/logistic/ $TOOLS/caffe train \
    --solver=models/sparsenet/logistic/solver_$1.prototxt \
    --weights=models/sparsenet/euclidean/sparsenet_v.34.0_iter_20000.caffemodel \
    $2
    #--gpu=1
