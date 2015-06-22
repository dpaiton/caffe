#!/usr/bin/env sh
# Compute the mean image from the mnist training lmdb
#

EXAMPLE=examples/mnist
DATA=data/mnist
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/mnist_train_lmdb \
  $DATA/mnist_mean.binaryproto

echo "Done."
