import caffe
import numpy as np
import json
from PIL import Image
from time import time
from cStringIO import StringIO

from os.path import join
import sys

from collections import defaultdict

np.random.seed(0)


from argparse import ArgumentParser
parser = ArgumentParser('sgd training driver')
parser.add_argument('-d', '--device_id', type=int, help='''gpu device number''',
                    default=-1)

model_file = '/src/caffe/models/sparsenet_mnist/sparsenet_iter_20000.caffemodel'
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

def main(args):
    if args.device_id == -1:
        print 'running on cpu'
        caffe.set_mode_cpu()
    else:
        device_id = int(args.device_id)
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

    net.forward()
    a = np.array(net.blobs['encode'].data)
    weights = np.array(net.params['decode'][0].data)
    biases = np.array(net.params['decode'][1].data)

    import IPython; IPython.embed()
   
    weight_vis = vis_square(weights.T.reshape(1000, 28, 28))
    
    weight_img = np.uint8(weight_vis*255)

    Image.fromarray(weight_img).save('/src/caffe/weights2.png')

    bias_vis = vis_square(biases.reshape(1, 28, 28))
    
    bias_img = np.uint8(bias_vis*255)

    Image.fromarray(bias_img).save('/src/caffe/bias2.png')




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
