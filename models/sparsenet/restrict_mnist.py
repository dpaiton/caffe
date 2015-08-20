import caffe
import numpy as np
import lmdb
import random

dataset_dir    = '/raid/dylan/mnist/'
#dataset_dir    = 'examples/mnist/'
orig_dataset   = dataset_dir+'/mnist_train_50K_lmdb/'
num_keep       = 100 
new_dataset    = dataset_dir+'/mnist_train_50K_lmdb_'+str(num_keep)+'/'
num_labels     = 10

assert num_keep%num_labels == 0

print 'Setting all but '+str(num_keep)+' images to label=ignore from dataset '+orig_dataset

orig_env = lmdb.open(orig_dataset, readonly=True)

# txn is a Transaction object
datum = caffe.proto.caffe_pb2.Datum()

# read once to get dim
with orig_env.begin() as txn:
    raw_datum = txn.get(b'00000000')

    datum.ParseFromString(raw_datum)

    channels = datum.channels
    height = datum.height
    width = datum.width

label_list = []

# loop through dataset once to get labels
with orig_env.begin() as txn:
    lmdb_cursor = txn.cursor()
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        label_list.append(datum.label)

dataset_size = len(label_list)
num_keep = int(num_keep/num_labels) # num_keep per label 

print 'Found '+str(dataset_size)+' images. Keeping '+str(num_keep)+' per label.'

idx_list = []
ignore_list = []
for num in range(0,10):
   idx_list.append([idx for idx,val in enumerate(label_list) if val==num])
   ignore_list.append(random.sample(idx_list[num],len(idx_list[num])-num_keep))

#TODO: *2 is not always enough - why?
num_bytes = np.prod((dataset_size,channels,height,width)) * np.dtype(np.uint8).itemsize * 8

mod_env = lmdb.open(new_dataset, map_size=num_bytes)

# write out modified dataset
num_ignored = 0
with mod_env.begin(write=True) as mod_txn:
    with orig_env.begin() as orig_txn:
        lmdb_cursor = orig_txn.cursor()
        iteration = 0
        for key, value in lmdb_cursor:
            datum.ParseFromString(value)
            if iteration in [item for sublist in ignore_list for item in sublist]:
                num_ignored += 1
                datum.label = int(-1)
            mod_txn.put(key, datum.SerializeToString())
            iteration += 1

print 'Set '+str(num_ignored)+' out of '+str(iteration)+' labels to ignore.'
