import os
import numpy as np
import config
import argparse
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
import tqdm
import glob
import time
import random
import json
from utils import *

class GNN_Detector(tf.keras.Model):
    def __init__(self, initial_channels,threshold_k,trainable = True):
        super(GNN_Detector, self).__init__(name='')
        self.threshold_k = threshold_k
        self.init_model(initial_channels, trainable)

    def init_model(self,initial_channels, trainable):
        with tf.name_scope("GNN_Detector_Model") as scope:
            self.graph_weight_1 = tf.Variable(tf.random.truncated_normal(shape=[initial_channels, config.GRAPH_CONV_LAYER_CHANNEL], stddev=0.1, dtype=tf.float32), trainable= trainable,name="graph_weight_1",)
            self.graph_weight_2 = tf.Variable(tf.random.truncated_normal(shape=[config.GRAPH_CONV_LAYER_CHANNEL, config.GRAPH_CONV_LAYER_CHANNEL], stddev=0.1, dtype=tf.float32), trainable= trainable,name="graph_weight_2")
            self.graph_weight_3 = tf.Variable(tf.random.truncated_normal(shape=[config.GRAPH_CONV_LAYER_CHANNEL, config.GRAPH_CONV_LAYER_CHANNEL], stddev=0.1, dtype=tf.float32), trainable= trainable,name="graph_weight_3")
            self.graph_weight_4 = tf.Variable(tf.random.truncated_normal(shape=[config.GRAPH_CONV_LAYER_CHANNEL, 1], stddev=0.1, dtype=tf.float32), trainable= False,name="graph_weight_4")
            self.conv1d_kernel_1 = tf.Variable(tf.random.truncated_normal(shape=[config.CONV1D_1_FILTER_WIDTH, 1, config.CONV1D_1_OUTPUT], stddev=0.1, dtype=tf.float32), trainable= trainable,name="conv1d_kernel_1",)
            self.conv1d_kernel_2 = tf.Variable(tf.random.truncated_normal(shape=[config.CONV1D_2_FILTER_WIDTH, config.CONV1D_1_OUTPUT, config.CONV1D_2_OUTPUT], stddev=0.1, dtype=tf.float32), trainable= trainable,name="conv1d_kernel_2",)
            self.weight_1 = tf.Variable(tf.random.truncated_normal(shape=[(self.threshold_k - config.CONV1D_2_FILTER_WIDTH + 1) *config.CONV1D_2_OUTPUT, config.DENSE_NODES], stddev=0.1 ,dtype=tf.float32), trainable= trainable,name="weight_1")
            self.bias_1 = tf.Variable(tf.zeros(shape=[config.DENSE_NODES],dtype=tf.float32), trainable= trainable,name="bias_1")
            self.weight_2 = tf.Variable(tf.random.truncated_normal(shape=[config.DENSE_NODES, config.num_classes],dtype=tf.float32), trainable= trainable,name="weight_2")
            self.bias_2 = tf.Variable(tf.zeros(shape=[config.num_classes],dtype=tf.float32), trainable= trainable,name="bias_2")
            self.opt = tf.optimizers.Adam(learning_rate=config.learning_rate, clipnorm=config.max_gradient_norm)

            self.flatten_layer = tf.keras.layers.Flatten()
            self.dropout_layer = tf.keras.layers.Dropout(config.dropout)

    def call(self, D_inverse, A_tilde, X, nodes_size_list, is_train, sparse = False):
        if sparse:
            if isinstance(D_inverse, list) == False:
                D_inverse = [D_inverse]
                A_tilde = [A_tilde]
                X = [X]
                nodes_size_list = [nodes_size_list]
            node_size_pl = nodes_size_list[0]
            D_inverse_pl = [tf.SparseTensor(tf.cast(_inverse[0],tf.int64),tf.cast(_inverse[1],tf.float32),_inverse[2]) for _inverse in D_inverse]
            A_tilde_pl = [tf.SparseTensor(tf.cast(_inverse[0],tf.int64),tf.cast(_inverse[1],tf.float32),_inverse[2]) for _inverse in A_tilde]
            X_pl = [tf.SparseTensor(tf.cast(_X[0],tf.int64),tf.cast(_X[1],tf.float32),_X[2]) for _X in X]
        else:  
            D_inverse_pl = [tf.cast(D_inverse.todense(), dtype=tf.float32)]
            A_tilde_pl = [tf.cast(A_tilde.todense(), dtype=tf.float32)]
            X_pl = [tf.cast(X, dtype=tf.float32)]
            node_size_pl = nodes_size_list
        is_train = is_train
        # GRAPH CONVOLUTION LAYER
        gl_1_XxW = dot(X_pl, self.graph_weight_1, sparse)
        gl_1_AxXxW = dot(A_tilde_pl, gl_1_XxW, sparse)
        Z_1 = tf.nn.tanh(dot(D_inverse_pl, gl_1_AxXxW, sparse))
        gl_2_XxW = tf.matmul(Z_1, self.graph_weight_2)
        gl_2_AxXxW = dot(A_tilde_pl, gl_2_XxW, sparse)
        Z_2 = tf.nn.tanh(dot(D_inverse_pl, gl_2_AxXxW, sparse))
        gl_3_XxW = tf.matmul(Z_2, self.graph_weight_3)
        gl_3_AxXxW = dot(A_tilde_pl, gl_3_XxW, sparse)
        Z_3 = tf.nn.tanh(dot(D_inverse_pl, gl_3_AxXxW, sparse))
        gl_4_XxW = tf.matmul(Z_3, self.graph_weight_4)
        gl_4_AxXxW = dot(A_tilde_pl, gl_4_XxW, sparse)
        Z_4 = tf.nn.tanh(dot(D_inverse_pl, gl_4_AxXxW, sparse))
        graph_conv_output = tf.concat([Z_1, Z_2, Z_3], axis=-1)  # shape=(node_size/None, 32+32+32)
        
        graph_conv_output_stored = [tf.gather(graph_conv_output[i], tf.nn.top_k(Z_4[i][:, 0], config.max_nodes).indices) for i in range(len(node_size_pl))]

        graph_conv_output_top_k = [tf.cond(tf.less(node_size_pl[i], self.threshold_k),
                                        lambda: tf.concat(axis=0,
                                                            values=[graph_conv_output_stored[i][:node_size_pl[i],:],
                                                                    tf.zeros(dtype=tf.float32,
                                                                            shape=[self.threshold_k-node_size_pl[i],
                                                                                    config.CONV1D_1_FILTER_WIDTH])]),
                                        lambda: tf.slice(graph_conv_output_stored[i], begin=[0, 0], size=[self.threshold_k, -1])) for i in range(len(node_size_pl))]
        
        # FLATTEN LAYER
        graph_conv_output_flatten = tf.reshape(graph_conv_output_top_k, shape=[len(node_size_pl), config.CONV1D_1_FILTER_WIDTH*self.threshold_k, 1])
        assert graph_conv_output_flatten.shape == [len(node_size_pl), config.CONV1D_1_FILTER_WIDTH*self.threshold_k, 1]

        # 1-D CONVOLUTION LAYER 1:
        # kernel = (filter_width, in_channel, out_channel)
        conv_1d_a = tf.nn.conv1d(graph_conv_output_flatten, self.conv1d_kernel_1, stride=config.CONV1D_1_FILTER_WIDTH, padding="VALID")
        assert conv_1d_a.shape == [len(node_size_pl), self.threshold_k, config.CONV1D_1_OUTPUT]
        # 1-D CONVOLUTION LAYER 2:
        conv_1d_b = tf.nn.conv1d(conv_1d_a, self.conv1d_kernel_2, stride=1, padding="VALID")
        assert conv_1d_b.shape == [len(node_size_pl), self.threshold_k - config.CONV1D_2_FILTER_WIDTH + 1, config.CONV1D_2_OUTPUT]
        conv_output_flatten = self.flatten_layer(conv_1d_b)

        # DENSE LAYER
        dense_z = tf.nn.relu(tf.matmul(conv_output_flatten, self.weight_1) + self.bias_1)
        #if tf.math.equal(is_train,True):
        #    dense_z = self.dropout_layer(dense_z)
        pre_y = tf.matmul(dense_z, self.weight_2) + self.bias_2
        pos_score = tf.nn.softmax(pre_y)
        return pos_score, pre_y, np.argmax(pre_y,-1)
    

    def cal_loss(self,label, logits):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)) 


import pdb
def train():
    if os.path.exists('input/'+config.current_malware+'opcode/') :
        ending = "*.json"
        train_graph_paths = glob.glob(config.train_graph_folder+ending)
        test_graph_paths = glob.glob(config.test_graph_folder+ending)
        print(config.test_graph_folder+ending)
    else:
        with open(config.data_lens, 'rb') as jf:
            train_graph_paths, val_graph_paths, test_graph_paths, lbs_train, lbs_val, lbs_test  = pickle.load(jf)
    #_, ckpt_manager = load_ckpt(model, 'models/gnn-mc/'+config.current_malware)
    model_name='models/gnn-mc/'+config.current_malware
    #config.top_k = 1000

    model = GNN_Detector(config.vector_size, config.top_k)
    _, ckpt_manager = load_ckpt(model, model_name)#, 'models/gnn-mc/final-3-bbr/ckpt-21')
    print(model_name)

    dics = read_all_data_proc(rewrite=False)
    #print(train_graph_paths)
    test_detector(model, ckpt_manager, 'gnn', dics, train_graph_paths, train_graph_paths,0)
    test_detector(model, ckpt_manager, 'gnn', dics, train_graph_paths, val_graph_paths, 0)
    test_detector(model, ckpt_manager, 'gnn', dics, train_graph_paths, test_graph_paths, 0)
    #exit()
    #train_detector_batch(model, ckpt_manager, 'gnn', dics, train_graph_paths, test_graph_paths, val_graph_paths)
    #show_node_importance(model, ckpt_manager, 'gnn', dics, train_graph_paths, test_graph_paths, 0)

def run(test_graph_paths,i):
    with open(config.data_lens+'-2gcn-'+str(i)+'.txt', 'rb') as jf:
        train_graph_paths, val_graph_paths, lbs_train, lbs_val  = pickle.load(jf)

    dics = read_all_data_proc(rewrite=False)#,path="malware/"+config.current_malware+config.current_feats+"gnn/all_data_adj_feature")

    model_name='models/gnn-mc-2-7-initial-4layers/'+str(i)+'/'+config.current_malware#+'add-ri-gene-30/'
    #model_name='models-1/gnn-mc/'+config.current_malware
    #config.top_k = 1000
    model = GNN_Detector(config.vector_size, config.top_k)
    _, ckpt_manager = load_ckpt(model, model_name)#, 'models-1/gnn-mc/'+config.current_malware+'ckpt-16')
    print(model_name)
    print("====")
    print('test set')
    _,_max = test_detector(model, ckpt_manager,  'gnn', dics, train_graph_paths, test_graph_paths, 0)
    print('train set')
    test_detector(model, ckpt_manager, 'gnn', dics, train_graph_paths, train_graph_paths, 0)
    print('val set')
    test_detector(model, ckpt_manager, 'gnn', dics, train_graph_paths, val_graph_paths, 0)
    #return
    config.learning_rate = 0.00300
    config.batch_size = 50
    config.epoch=10
    _max= train_detector_batch(model, ckpt_manager, 'gnn', dics, train_graph_paths, test_graph_paths, val_graph_paths, _max)
    config.learning_rate = 0.00300
    config.batch_size = 50
    config.epoch=10
    _max= train_detector_batch(model, ckpt_manager, 'gnn', dics, train_graph_paths, test_graph_paths, val_graph_paths, _max)
    test_detector(model, ckpt_manager, 'gnn', dics, train_graph_paths, train_graph_paths,0)
    test_detector(model, ckpt_manager, 'gnn', dics, train_graph_paths, test_graph_paths, 0)
    test_detector(model, ckpt_manager, 'gnn', dics, train_graph_paths, val_graph_paths, 0)

from multiprocessing import Process
def cross():
    with open(config.data_lens+'-all.txt', 'rb') as jf:
        train_graph_paths, val_graph_paths, test_graph_paths, lbs_train, lbs_val, lbs_test  = pickle.load(jf)
    
    config.learning_rate = 0.00300
    config.batch_size = 50
    config.epoch=30
    for i in range(4,6):
        #p = Process(target=run, args=(test_graph_paths,i,))
        #p.start()
        run(test_graph_paths, i)
    

if __name__ == '__main__':
    cross()
