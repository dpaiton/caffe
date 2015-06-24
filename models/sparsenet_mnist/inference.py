import caffe
import numpy as np
import json
from PIL import Image
from time import time
from cStringIO import StringIO
from matplotlib import pyplot as plt
from os.path import join
import sys
import IPython

from collections import defaultdict

np.random.seed(0)


from argparse import ArgumentParser
parser = ArgumentParser('sgd training driver')
parser.add_argument('-d', '--device_id', type=int, help='''gpu device number''',
                    default=-1)

root_dir   = '/osx/caffe/'
model_ver  = 'v.0.5'
max_iter   = 1598000
model_file = root_dir+'/models/sparsenet_mnist/sparsenet_'+model_ver+'_iter_'+str(max_iter)+'.caffemodel'
model_prototxt = 'models/sparsenet_mnist/sparsenet.prototxt'

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    return data

def make_movies(start,end,step):
    for iter in range(start,end,step):
    	model_file = root_dir+'/models/sparsenet_mnist/sparsenet_'+model_ver+'_iter_'+str(iter)+'.caffemodel'
    	net        = caffe.Net(model_prototxt, model_file, caffe.TEST)

    	weights    = np.array(net.params['decode'][0].data)
        weight_vis = vis_square(weights.T.reshape(weights.shape[1], 28, 28))
        weight_img = np.uint8(weight_vis*255)
        Image.fromarray(weight_img).save(root_dir+'Analysis/weights_'+model_ver+'_'+str(iter)+'.png')
	
	weights_l2 = np.sqrt(np.sum(weights**2,axis=0))
	plt.bar(np.arange(0,len(weights_l2)),weights_l2)
	plt.savefig(root_dir+'Analysis/weight_l2_'+model_ver+'_'+str(iter)+'.png',bbox_inches='tight')
	plt.clf()

	biases   = np.array(net.params['decode'][1].data)
	bias_vis = vis_square(biases.reshape(1, 28, 28))
	bias_img = np.uint8(bias_vis*255)
	Image.fromarray(bias_img).save(root_dir+'Analysis/bias_'+model_ver+'_'+str(iter)+'.png')

        net.forward()
	activity = np.array(net.blobs['encode'].data)
        activity_img = activity / np.max(np.abs(activity)) * 255./2 + 255./2
	activity_img = np.uint8(activity_img)
	Image.fromarray(activity_img).save(root_dir+'Analysis/activity_'+model_ver+'_'+str(iter)+'.png')

	print "Iteration # "+str(iter)

	#IPython.embed()

def main(args):
    if args.device_id == -1:
        print 'running on cpu'
        caffe.set_mode_cpu()
    else:
        print 'running on gpu %d' % device_id
        caffe.set_device(device_id)
        caffe.set_mode_gpu()

    net = caffe.Net(model_prototxt, model_file, caffe.TEST)

    batch_size  = net.blobs['data'].data.shape[0]
    batch_shape = net.blobs['data'].data.shape[1:]

    # each output is (batch size, feature dim, spatial dim)
    print [(k, v.data.shape) for k, v in net.blobs.items()]

    # just print the weight sizes (not biases)
    print [(k, v[0].data.shape) for k, v in net.params.items()]

    net.forward()
    input_dat = np.squeeze(np.array(net.blobs['data'].data))
    recon     = np.array(net.blobs['decode'].data).reshape(input_dat.shape)

    input_vis = vis_square(input_dat)
    input_img = np.uint8(input_vis*255)
    Image.fromarray(input_img).save(root_dir+'Analysis/input_img_'+model_ver+'.png')

    recon_vis = vis_square(recon)
    recon_img = np.uint8(recon_vis*255)
    Image.fromarray(recon_img).save(root_dir+'Analysis/recon_'+model_ver+'.png')

    make_movies(1000,max_iter+100,1000)

    activity = []
    activity.append(np.array(net.blobs['encode'].data))
    for iter in range(50):
        net.forward()
        activity.append(np.array(net.blobs['encode'].data))
    plt.hist(np.vstack(activity).flatten(),bins=1000)
    plt.savefig(root_dir+'Analysis/activity_hist_'+model_ver+'.png',bbox_inches='tight')

    #IPython.embed()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
