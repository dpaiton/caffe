import caffe
import numpy as np
import lmdb
import random

dataset_dir    = 'examples/mnist/'
num_ignore     = 100
new_dataset    = dataset_dir+'/mnist_train_lmdb_'+str(num_ignore)+'/'

env = lmdb.open(new_dataset, readonly=True)

# txn is a Transaction object
datum = caffe.proto.caffe_pb2.Datum()

# read once to get dim
label_list = []

# loop through dataset once to get labels
with env.begin() as txn:
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
listIg = [idx for idx,val in enumerate(label_list) if val==-1]

print '-1: '+str(len(listIg))
print '0: '+str(len(list0))
print '1: '+str(len(list1))
print '2: '+str(len(list2))
print '3: '+str(len(list3))
print '4: '+str(len(list4))
print '5: '+str(len(list5))
print '6: '+str(len(list6))
print '7: '+str(len(list7))
print '8: '+str(len(list8))
print '9: '+str(len(list9))
print 'total: '+str(len(listIg)+len(list0)+len(list1)+len(list2)+len(list3)+len(list4)+len(list5)+len(list6)+len(list7)+len(list8)+len(list9))
