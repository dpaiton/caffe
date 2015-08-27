import caffe
import numpy as np
import os
from argparse import ArgumentParser
import re
import matplotlib.pyplot as plt
import IPython

np.random.seed(0)

parser = ArgumentParser('sgd training driver')
parser.add_argument('-d', '--device_id', type=int, help='''gpu device number''',
                    default=-1)

root_dir   = '/Users/dpaiton/Code/caffe/'
exp_lbl    = 'logistic'  # logistic or euclidean
model_lbl  = 'sparsenet' # sparsenet or mlp
model_ver  = 'v.100.8'
mov_start  = 1000
mov_step   = 1000
mov_end    = 754000

model_pretxt    = root_dir+'/models/sparsenet/'+exp_lbl+'/checkpoints/'+model_lbl+'_'+model_ver+'_iter_'
model_prototxt  = root_dir+'/models/sparsenet/'+exp_lbl+'/'+model_lbl+'.prototxt'

if exp_lbl is 'logistic':
    solver_prototxt = root_dir+'/models/sparsenet/'+exp_lbl+'/solver_sparse.prototxt'
elif exp_lbl is 'euclidean':
    solver_prototxt = root_dir+'/models/sparsenet/'+exp_lbl+'/solver.prototxt'
else:
    raise ValueError('exp_lbl must be either "logistic" or "euclidean"')

out_dir = root_dir+'/models/sparsenet/'+exp_lbl+'/Analysis/'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def main(args):
    if args.device_id == -1:
        print 'running on cpu'
        caffe.set_mode_cpu()
    else:
        print 'running on gpu %d' % device_id
        caffe.set_device(device_id)
        caffe.set_mode_gpu()

    # time steps
    model_times = np.arange(mov_start,mov_end+1,mov_step)

    # get lambda value
    if exp_lbl is 'logistic':
        num_labels = re.findall('v\.(\d+)\.',model_ver)[0]
    text = open(model_prototxt,'r').read()
    lamb = float(re.findall("lambda\: (\d+.\d+)", text)[0])
    batch_size = float(re.findall("batch_size: (\d+)",text)[0])

    accuracy_vals  = np.zeros((len(model_times)))
    energy_vals    = np.zeros((len(model_times)))

    # prepare solver
    solver = caffe.get_solver(solver_prototxt)

    index = 0
    for model_itr in model_times:
        model_file = model_pretxt+str(model_itr)+'.caffemodel'
        solver.net.copy_from(model_file)
        solver.step(1)

        scores_train = solver.net.forward()
        energy_vals[index] += 0.9*scores_train['softmax_loss'] + 0.1*scores_train['euclidean_loss']

        for i in range(100):
            scores_test  = solver.test_nets[0].forward()
            accuracy_vals[index] += scores_test['accuracy']

        index += 1

    fig, ax1 = plt.subplots()
    ax1.plot(model_times, energy_vals, 'b',label='energy')
    ax1.set_ylabel('Energy')
    ax1.set_xlabel('Model Time Step')
    ax2 = ax1.twinx()
    ax2.plot(model_times,accuracy_vals,'r',label='accuracy')
    ax2.set_ylabel('Accuracy')
    if exp_lbl is 'logistic':
        plt.title('Energy analysis for '+num_labels+' training labels')
    else:
        plt.title('Energy analysis')
    legend = fig.legend(loc='best', shadow=False, fontsize='small')
    plt.savefig(out_dir+'/energy_'+model_ver+'.png',dpi=1000,bbox_inches='tight')

    #IPython.embed()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
