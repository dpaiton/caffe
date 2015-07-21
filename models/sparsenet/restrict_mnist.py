import caffe
import numpy as np
import lmdb
import random

dataset_dir    = 'examples/mnist/'
orig_dataset   = dataset_dir+'/mnist_train_lmdb/'
percent_ignore = 10
new_dataset    = dataset_dir+'/mnist_train_lmdb_'+str(percent_ignore)+'/'

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

list0 = [idx for idx,val in enumerate(label_list) if val==0]
list1 = [idx for idx,val in enumerate(label_list) if val==1]
list2 = [idx for idx,val in enumerate(label_list) if val==2]
list3 = [idx for idx,val in enumerate(label_list) if val==3]
list4 = [idx for idx,val in enumerate(label_list) if val==4]
list5 = [idx for idx,val in enumerate(label_list) if val==5]
list6 = [idx for idx,val in enumerate(label_list) if val==6]
list7 = [idx for idx,val in enumerate(label_list) if val==7]
list8 = [idx for idx,val in enumerate(label_list) if val==8]
list9 = [idx for idx,val in enumerate(label_list) if val==9]

ign0 = random.sample(list0,int(np.floor(len(list0)*(percent_ignore/100))))
ign1 = random.sample(list1,int(np.floor(len(list1)*(percent_ignore/100))))
ign2 = random.sample(list2,int(np.floor(len(list2)*(percent_ignore/100))))
ign3 = random.sample(list3,int(np.floor(len(list3)*(percent_ignore/100))))
ign4 = random.sample(list4,int(np.floor(len(list4)*(percent_ignore/100))))
ign5 = random.sample(list5,int(np.floor(len(list5)*(percent_ignore/100))))
ign6 = random.sample(list6,int(np.floor(len(list6)*(percent_ignore/100))))
ign7 = random.sample(list7,int(np.floor(len(list7)*(percent_ignore/100))))
ign8 = random.sample(list8,int(np.floor(len(list8)*(percent_ignore/100))))
ign9 = random.sample(list9,int(np.floor(len(list9)*(percent_ignore/100))))

ignore_list = np.concatenate((ign0,ign1,ign2,ign3,ign4,ign5,ign6,ign7,ign8,ign9))

num_bytes = np.prod((dataset_size,channels,height,width)) * np.dtype(np.uint8).itemsize * 2

mod_env = lmdb.open(new_dataset, map_size=num_bytes)

# write out modified dataset
num_ignore = 0
with mod_env.begin(write=True) as mod_txn:
    with orig_env.begin() as orig_txn:
        lmdb_cursor = orig_txn.cursor()
        iteration = 0
        for key, value in lmdb_cursor:
            datum.ParseFromString(value)
            if iteration in ignore_list:
                num_ignore += 1
                datum.label = int(-1)
            mod_txn.put(key, datum.SerializeToString())
            iteration += 1

print 'Set '+str(num_ignore)+' out of '+str(dataset_size)+' labels to ignore.'
