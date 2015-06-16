import caffe
import numpy as np
import json
from PIL import Image
from time import time
from cStringIO import StringIO
from matplotlib import pyplot as plt
from os.path import join
import sys

from collections import defaultdict

np.random.seed(0)


from argparse import ArgumentParser
parser = ArgumentParser('sgd training driver')
parser.add_argument('-d', '--device_id', type=int, help='''gpu device number''',
                    default=-1)

model_file = '/caffe/models/sparsenet_mnist/sparsenet_iter_500.caffemodel'
model_prototxt = '/caffe/models/sparsenet_mnist/sparsenet.prototxt'

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

def main(args):
    if args.device_id == -1:
        print 'running on cpu'
        caffe.set_mode_cpu()
    else:
        print 'running on gpu %d' % device_id
        caffe.set_device(device_id)
        caffe.set_mode_gpu()

    net = caffe.Net(model_prototxt, model_file, caffe.TEST)

    batch_size = net.blobs['data'].data.shape[0]
    batch_shape = net.blobs['data'].data.shape[1:]

    # each output is (batch size, feature dim, spatial dim)
    print [(k, v.data.shape) for k, v in net.blobs.items()]

    # just print the weight sizes (not biases)
    print [(k, v[0].data.shape) for k, v in net.params.items()]

    # TODO: Don't index batch # 0, but instead use vis_square to see all batches
    net.forward()
    input     = np.array(net.blobs['data'].data)
    activity  = np.array(net.blobs['encode'].data[0])
    weights   = np.array(net.params['decode'][0].data)
    biases    = np.array(net.params['decode'][1].data)
    recon     = np.array(net.blobs['decode'].data).reshape(input_img.shape)

    import IPython; IPython.embed()

    input_vis = vis_square(input)
    input_img = np.uint8(input_vis*255)
    Image.fromarray(input_img).save('/caffe/Analysis/input_img.png')

    weight_vis = vis_square(weights.T.reshape(weights.shape[1], input_img.shape[0], input_img.shape[1]))
    weight_img = np.uint8(weight_vis*255)
    Image.fromarray(weight_img).save('/caffe/Analysis/weights.png')

    bias_vis = vis_square(biases.reshape(1, input_img.shape[0], input_img.shape[1]))
    bias_img = np.uint8(bias_vis*255)
    Image.fromarray(bias_img).save('/caffe/Analysis/bias.png')

    recon_vis = vis_square(recon)
    recon_img = np.uint8(recon_vis*255)
    Image.fromarray(recon_img).save('/caffe/Analysis/recon.png')

    plt.hist(activity,bins=activity.shape[0]/10)
    plt.savefig('/caffe/Analysis/activity.png',bbox_inches='tight')
 
    mean_recon_error = np.mean(np.sqrt((input_img-recon)**2))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
