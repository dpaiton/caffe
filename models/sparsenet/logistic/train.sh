#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=models/sparsenet/logistic/ $TOOLS/caffe train \
    --solver=models/sparsenet/logistic/solver_$1.prototxt \
    $2
    #--weights=models/sparsenet/euclidean/sparsenet_v.33.0_iter_2000000.caffemodel \
    #--gpu=1
