import caffe
import numpy as np
import lmdb
import random

#dataset_dir      = '/raid/dylan/mnist/'
dataset_dir      = 'examples/mnist/'
num_val          = 10000
num_tra          = 50000
orig_tra_dataset = dataset_dir+'/mnist_train_lmdb/'
tra_dataset      = dataset_dir+'/mnist_train_'+str(num_tra/1000)+'K_lmdb/'
val_dataset      = dataset_dir+'/mnist_val_'+str(num_val/1000)+'K_lmdb/'

print 'Creating validation set with '+str(num_val)+' images.'

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

num_val_bytes = np.prod((num_val,channels,height,width)) * np.dtype(np.uint8).itemsize * 8
val_env = lmdb.open(val_dataset, map_size=num_val_bytes)

# randomly select images to place into the validation set
val_idx = random.sample([idx for idx in range(0,num_val+num_tra)], num_val)

val_inc = 0
tra_inc = 0
# write out modified dataset
with tra_env.begin(write=True) as tra_txn:
    with val_env.begin(write=True) as val_txn:
        with orig_env.begin() as orig_txn:
            lmdb_cursor = orig_txn.cursor()
            iteration = 0
            for key, value in lmdb_cursor:
                datum.ParseFromString(value)
                if iteration in val_idx:
                    val_txn.put(key, datum.SerializeToString())
                    val_inc += 1
                else:
                    tra_txn.put(key, datum.SerializeToString())
                    tra_inc += 1
                iteration += 1

assert val_inc == num_val
assert tra_inc == num_tra
print 'From '+str(iteration)+' images, used '+str(tra_inc)+' for training and '+str(val_inc)+' for validation.'
