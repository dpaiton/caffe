import caffe
import matplotlib.pyplot as plt
import numpy as np

caffe_root     = '/Users/dpaiton/Code/caffe/'
model_list     = ['sparsenet']#['sparsenet', 'mlp']
version_list   = [90]
iteration_list = range(10000,240001,10000)
model_num      = 15

iteration      = iteration_list[-1]

caffe.set_mode_cpu()

lbl_err = np.zeros((len(model_list),len(version_list)))

for midx, model in enumerate(model_list):
    for vidx, version in enumerate(version_list):
        model_file = caffe_root+'models/sparsenet/logistic/checkpoints/'+model+ \
                    '_v.'+str(version)+'.'+str(model_num)+'_iter_'+str(iteration)+'.caffemodel'

        net_proto  = caffe_root+'models/sparsenet/logistic/'+model+'.prototxt'

        net = caffe.Net(net_proto, model_file, caffe.TEST)

        lbl_accuracy = 0

        for iter in range(100):
            out = net.forward()
            lbl_accuracy += out['accuracy']

        lbl_accuracy /= 100.

        lbl_err[midx,vidx] = 1 - lbl_accuracy

version = version_list[-1]

itr_err = np.zeros((len(model_list),len(iteration_list)))

for midx, model in enumerate(model_list):
    for iidx, iteration in enumerate(iteration_list):
        model_file = caffe_root+'models/sparsenet/logistic/checkpoints/'+model+ \
                    '_v.'+str(version)+'.'+str(model_num)+'_iter_'+str(iteration)+'.caffemodel'

        net_proto  = caffe_root+'models/sparsenet/logistic/'+model+'.prototxt'

        net = caffe.Net(net_proto, model_file, caffe.TEST)

        itr_accuracy = 0

        for iter in range(100):
            out = net.forward()
            itr_accuracy += out['accuracy']

        itr_accuracy /= 100.

        itr_err[midx,iidx] = 1 - itr_accuracy


print model_list[0]+" lbl err %: "+str(100*lbl_err[0,:])
print model_list[0]+" itr err %: "+str(100*itr_err[0,:])
print "\n"
if len(model_list) > 1:
    print model_list[1]+" lbl err %: "+str(100*lbl_err[1,:])
    print model_list[1]+" itr err %: "+str(100*itr_err[1,:])
print "Analysis of model number "+str(model_num)+" complete."

plt.figure(1)
plt.plot(iteration_list, 100*itr_err[0,:], 'r', label=model_list[0])
if len(model_list) > 1:
    plt.plot(iteration_list, 100*itr_err[1,:], 'b', label=model_list[1])
plt.xlabel('Iteration #')
plt.ylabel('% Error Rate')
plt.title('Error Rate with '+str(100-version)+' % Labels')
plt.legend(loc='upper left')#bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

if len(version_list) > 1:
    plt.figure(2)
    plt.plot(version_list, 100*lbl_err[0,:], 'r', label=model_list[0])
    if len(model_list) > 1:
        plt.plot(version_list, 100*lbl_err[1,:], 'b', label=model_list[1])
    plt.xlabel('% of Labels Set to Ignore')
    plt.ylabel('% Error Rate')
    plt.title('Error Rate with Diminishing Labels')
    plt.legend(loc='upper left')#bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
