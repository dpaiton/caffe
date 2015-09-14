#import matplotlib
#matplotlib.use('Agg')
import caffe
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
import IPython

np.random.seed(0)

root_dir   = '/Users/dpaiton/Code/caffe/'
#root_dir   = '/nfs/dylan/caffe/'
exp_lbl    = 'logistic'  # logistic or euclidean
model_lbl  = 'mlp' # sparsenet or mlp
model_ver  = 'v.test'

def main():

    print 'Ploting data for '+exp_lbl+'_'+model_lbl+'_'+model_ver

    out_dir = root_dir+'/models/sparsenet/'+exp_lbl+'/Analysis/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    log_dir = root_dir+'models/sparsenet/'+exp_lbl+'/logfiles/'+model_lbl+'*'
    files = glob.glob(log_dir)
    log_file = [fi for fi in files if model_ver in fi][0]

    log_text = open(log_file,'r').read()

    max_iter = float(re.findall("max_iter: (\d+)",log_text)[0])

    test_start = 0
    test_step = float(re.findall("test_interval: (\d+)",log_text)[0])
    test_times = np.arange(test_start,max_iter+1,test_step)#[1:]

    train_start = 0
    train_step = float(re.findall("display: (\d+)",log_text)[0])
    train_times = np.arange(train_start,max_iter,train_step)#[1:]

    loss_types = re.findall('type\: \"(\w+)Loss\"', log_text)
    loss_vals = [float(val) for val in re.findall('loss_weight\: (\d+\.?\d*)', log_text)]
    if not loss_vals: #it's possible that it wasn't specified
        loss_vals = [1.0 for type in loss_types]

    loss_pairs = dict(zip(loss_types,loss_vals))

    tmp_list = [float(val) for val in re.findall('Train net output \#\d\: euclidean_loss \= (\d+\.?\d*e?-?\d*)',log_text)]#[1:]]
    euclidean_list = tmp_list if tmp_list else [0.0 for time in train_times]
    tmp_list = [float(val) for val in re.findall('Train net output \#\d\: softmax_loss \= (\d+\.?\d*e?-?\d*)',log_text)]#[1:]]
    softmax_list = tmp_list if tmp_list else [0.0 for time in train_times]
    tmp_list = [float(val) for val in re.findall('lr \= (\d+\.?\d*)',log_text)]#[1:]]
    lr_list = tmp_list if tmp_list else [0.0 for time in train_times]

    soft_mult = 1.0
    euc_mult = 1.0
    if 'SoftmaxWith' in loss_pairs.keys():
        soft_mult = loss_pairs['SoftmaxWith']
    if 'Euclidean' in loss_pairs.keys():
        euc_mult = loss_pairs['Euclidean']

    energy_vals = np.array([euc_mult*a + soft_mult*b for a,b in zip(euclidean_list,softmax_list)])
    accuracy_vals = np.array([float(val) for val in re.findall('Test net output \#\d\: accuracy \= (\d+\.?\d*)',log_text)])#[1:]])
    lr_vals = np.array(lr_list)

    if np.max(energy_vals) > 0:
        energy_vals /= np.max(energy_vals)
    if np.max(lr_vals) > 0:
        lr_vals /= np.max(lr_vals)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    line1 = ax1.plot(train_times, energy_vals, 'b',label='energy')
    ax1.set_ylabel('Energy',color='b')
    ax1.set_xlabel('Model Time Step')
    ax1.set_ylim([0,1])

    ax2 = ax1.twinx()
    line2 = ax2.plot(test_times,accuracy_vals,'r',label='accuracy')
    ax2.set_ylabel('Accuracy',color='r')
    ax2.set_ylim([0,1])

    ax3 = ax1.twinx()
    line3 = ax3.plot(train_times, lr_vals, 'g',label='learning rate')
    ax3.axes.get_yaxis().set_visible(False)

    if exp_lbl is 'logistic':
        num_labels = re.findall('v\.(\w+)',model_ver)[0]
        plt.title('Energy analysis for '+num_labels+' training labels')
    else:
        plt.title('Energy analysis')

    lines = line1 + line2 + line3
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, shadow=False, fontsize='small', bbox_to_anchor=(1.31,1))
    fig1.savefig(out_dir+'/energy_'+model_lbl+'_'+model_ver+'.png',dpi=200,bbox_inches='tight')

    IPython.embed()

if __name__ == '__main__':
    main()
