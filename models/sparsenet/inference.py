import caffe
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import IPython
from argparse import ArgumentParser

np.random.seed(0)

parser = ArgumentParser('sgd training driver')
parser.add_argument('-d', '--device_id', type=int, help='''gpu device number''',
                    default=-1)

root_dir   = '/Users/dpaiton/Code/caffe/'
exp_lbl    = 'euclidean'  # logistic or euclidean
model_lbl  = 'sparsenet' # sparsenet or mlp
model_ver  = 'v.33.0'
mov_start  = 2000000
mov_step   = 10000
mov_end    = 2000000

assert mov_start <= mov_end

#weight_layer_name = 'ip1'
weight_layer_name = 'encode'
activity_analysis = False 
pixel_bias        = False 
make_recon        = False 

model_pretxt  = root_dir+'/models/sparsenet/'+exp_lbl+'/checkpoints/'+model_lbl+'_'+model_ver+'_iter_'
model_file     = model_pretxt+str(mov_end)+'.caffemodel'
model_prototxt = 'models/sparsenet/'+exp_lbl+'/'+model_lbl+'.prototxt'
out_dir        = root_dir+'/models/sparsenet/'+exp_lbl+'/Analysis/'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

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
    	model_file = model_pretxt+str(iter)+'.caffemodel'

        #IPython.embed()
    	net        = caffe.Net(root_dir+model_prototxt, model_file, caffe.TEST)

    	weights    = np.array(net.params[weight_layer_name][0].data)

        #TODO: the general form doesn't work because the MLP IP layer is T from Sparsenet Encode Layer
        #      read input layer information to get pixel dims, other dim will be # weights
        #weight_len = np.int32(np.sqrt(weights.shape[0]))
        #weight_vis = vis_square(weights.T.reshape(weights.shape[1], weight_len, weight_len))

        #TODO: Don't hard code this
        weight_len = 28 
        weight_vis = vis_square(weights.T.reshape(500, weight_len, weight_len)) # for sparsenet
        #weight_vis = vis_square(weights.reshape(500, weight_len, weight_len))    # for MLP

        weight_img = np.uint8(weight_vis*255)
        Image.fromarray(weight_img).save(out_dir+'/'+weight_layer_name+'_weights_'+model_ver+'_'+str(iter)+'.png')
	
    weights_l2 = np.sqrt(np.sum(weights**2,axis=0))
    plt.bar(np.arange(0,len(weights_l2)),weights_l2)
    plt.savefig(out_dir+'/weight_l2_'+model_ver+'_'+str(iter)+'.png',bbox_inches='tight')
    plt.clf()

    if pixel_bias:
        biases   = np.array(net.params[weight_layer_name][1].data)
        bias_len = np.int32(np.sqrt(biases.shape[0]))
        bias_vis = vis_square(biases.reshape(1, bias_len, bias_len))
        bias_img = np.uint8(bias_vis*255)
        Image.fromarray(bias_img).save(out_dir+'/'+weight_layer_name+'_bias_'+model_ver+'_'+str(iter)+'.png')

    if activity_analysis:
        net.forward()
        activity = np.array(net.blobs['encode'].data)
        activity_img = activity / np.max(np.abs(activity)) * 255./2 + 255./2
        activity_img = np.uint8(activity_img)
        Image.fromarray(activity_img).save(out_dir+'/activity_'+model_ver+'_'+str(iter)+'.png')

def main(args):
    if args.device_id == -1:
        print 'running on cpu'
        caffe.set_mode_cpu()
    else:
        print 'running on gpu %d' % device_id
        caffe.set_device(device_id)
        caffe.set_mode_gpu()

    make_movies(mov_start,mov_end+1,mov_step)

    net = caffe.Net(root_dir+model_prototxt, model_file, caffe.TEST)

    batch_size  = net.blobs['data'].data.shape[0]
    batch_shape = net.blobs['data'].data.shape[1:]

    # each output is (batch size, feature dim, spatial dim)
    print [(k, v.data.shape) for k, v in net.blobs.items()]

    # print the weight sizes
    print [(k, v[0].data.shape) for k, v in net.params.items()]

    net.forward()
    input_dat = np.squeeze(np.array(net.blobs['data'].data))
    input_vis = vis_square(input_dat)
    input_img = np.uint8(input_vis*255)
    Image.fromarray(input_img).save(out_dir+'/input_img_'+model_ver+'.png')

    if make_recon:
        recon     = np.array(net.blobs[weight_layer_name].data).reshape(input_dat.shape)
        recon_vis = vis_square(recon)
        recon_img = np.uint8(recon_vis*255)
        Image.fromarray(recon_img).save(out_dir+'/recon_'+model_ver+'.png')


    if activity_analysis:
        activity = []
        activity.append(np.array(net.blobs['encode'].data))
        for iter in range(50):
            net.forward()
            activity.append(np.array(net.blobs['encode'].data))
        plt.hist(np.vstack(activity).flatten(),bins=1000)
        plt.savefig(out_dir+'/activity_hist_'+model_ver+'.png',bbox_inches='tight')

    #IPython.embed()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
