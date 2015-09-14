import caffe
import numpy as np
import lmdb
import random

#dataset_dir      = '/raid/dylan/mnist/'
dataset_dir      = 'examples/mnist/'
num_tra          = 100
orig_tra_dataset = dataset_dir+'/mnist_train_lmdb/'
tra_dataset      = dataset_dir+'/mnist_train_'+str(num_tra)+'_lmdb/'

num_labels = 10
assert num_tra%num_labels == 0

print 'Creating training set with '+str(num_tra)+' images.'

orig_env = lmdb.open(orig_tra_dataset, readonly=True)

# txn is a Transaction object
datum = caffe.proto.caffe_pb2.Datum()

# read once to get dim
with orig_env.begin() as txn:
    raw_datum = txn.get(b'00000000')

    datum.ParseFromString(raw_datum)

    channels = datum.channels
    height   = datum.height
    width    = datum.width

#TODO: *2 is not always enough - why?
num_tra_bytes = np.prod((num_tra,channels,height,width)) * np.dtype(np.uint8).itemsize * 8
tra_env = lmdb.open(tra_dataset, map_size=num_tra_bytes)

label_list = []
# loop through dataset to get labels
with orig_env.begin() as orig_txn:
    lmdb_cursor = orig_txn.cursor()
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        label_list.append(datum.label)

idx_list = []
keep_list = []
for num in range(0,num_labels):
   idx_list.append([idx for idx,val in enumerate(label_list) if val==num])
   keep_list.append(random.sample(idx_list[num],num_tra/num_labels))

num_kept = 0
with orig_env.begin() as orig_txn:
    with tra_env.begin(write=True) as tra_txn:
        lmdb_cursor = orig_txn.cursor()
        iteration = 0
        for key, value in lmdb_cursor:
            datum.ParseFromString(value)
            if iteration in [item for sublist in keep_list for item in sublist]:
                num_kept += 1
                tra_txn.put(key, datum.SerializeToString())
            iteration += 1

print 'From '+str(iteration)+' images, used '+str(num_kept)+' for training.'
