import caffe
import numpy as np
import lmdb
import re

new_dataset = 'examples/mnist/mnist_train_100_lmdb/'

env = lmdb.open(new_dataset, readonly=True)

# txn is a Transaction object
datum = caffe.proto.caffe_pb2.Datum()

# loop through dataset once to get labels
label_list = [str(x) for x in range(-1,10)]
counter = np.zeros(11)
with env.begin() as txn:
    lmdb_cursor = txn.cursor()
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        counter[datum.label+1] += 1

dataset_size = sum(counter)

print('\n'.join('{}: {}'.format(*k) for k in zip(label_list,counter)))
print 'total: ' + str(dataset_size)
