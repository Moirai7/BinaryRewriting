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
from  layers import ConvolutionalLayer
from utils import *
from sklearn.model_selection import KFold
from tensorflow.keras import layers
tf.keras.backend.set_floatx('float32')
class GNN_Detector(tf.keras.Model):
    def __init__(self, initial_channels,threshold_k,trainable = True):
        super(GNN_Detector, self).__init__(name='')
        self.threshold_k = threshold_k
        self.init_model(initial_channels, trainable)

    def init_model(self,initial_channels, trainable):
        with tf.name_scope("GNN_Detector_Model") as scope:
            self.normalization_layer = layers.experimental.preprocessing.Rescaling(1./257)
            #'''
            self.hidden_layer1 = ConvolutionalLayer(input_dim=initial_channels,
                                                output_dim=config.GRAPH_CONV_LAYER_CHANNEL,
                                                activation=tf.nn.tanh,
                                                sparse_inputs=True,
                                                bias=False,
                                                featureless = False,
                                                trainable = trainable)
            self.hidden_layer2 = ConvolutionalLayer(input_dim=config.GRAPH_CONV_LAYER_CHANNEL,
                                                output_dim=config.GRAPH_CONV_LAYER_CHANNEL,
                                                activation=tf.nn.tanh,
                                                sparse_inputs=False,
                                                bias=False,
                                                featureless = False,
                                                trainable = trainable)
            self.hidden_layer3 = ConvolutionalLayer(input_dim=config.GRAPH_CONV_LAYER_CHANNEL,
                                                output_dim=config.GRAPH_CONV_LAYER_CHANNEL,
                                                activation=tf.nn.tanh,
                                                sparse_inputs=False,
                                                bias=False,
                                                featureless = False,
                                                trainable = trainable)
            self.hidden_layer4 = ConvolutionalLayer(input_dim=config.GRAPH_CONV_LAYER_CHANNEL,
                                                output_dim=config.GRAPH_CONV_LAYER_CHANNEL,
                                                activation=tf.nn.tanh,
                                                sparse_inputs=False,
                                                bias=False,
                                                featureless = False,
                                                trainable = trainable)
            '''
            self.graph_weight_1 = tf.Variable(tf.random.truncated_normal(shape=[initial_channels, config.GRAPH_CONV_LAYER_CHANNEL], stddev=0.1, dtype=tf.float32), trainable= trainable,name="graph_weight_1",)
            self.graph_weight_2 = tf.Variable(tf.random.truncated_normal(shape=[config.GRAPH_CONV_LAYER_CHANNEL, config.GRAPH_CONV_LAYER_CHANNEL], stddev=0.1, dtype=tf.float32), trainable= trainable,name="graph_weight_2")
            self.graph_weight_3 = tf.Variable(tf.random.truncated_normal(shape=[config.GRAPH_CONV_LAYER_CHANNEL, config.GRAPH_CONV_LAYER_CHANNEL], stddev=0.1, dtype=tf.float32), trainable= trainable,name="graph_weight_3")
            self.graph_weight_4 = tf.Variable(tf.random.truncated_normal(shape=[config.GRAPH_CONV_LAYER_CHANNEL, 1], stddev=0.1, dtype=tf.float32), trainable= False,name="graph_weight_4")
            
            '''
            self.dim = config.GRAPH_CONV_LAYER_CHANNEL * 4#
            '''
            self.conv1d_kernel_1 = tf.Variable(tf.random.truncated_normal(shape=[self.dim, 1, config.CONV1D_1_OUTPUT], stddev=0.1, dtype=tf.float32), trainable= trainable,name="conv1d_kernel_1",)
            self.conv1d_kernel_2 = tf.Variable(tf.random.truncated_normal(shape=[config.CONV1D_2_FILTER_WIDTH, config.CONV1D_1_OUTPUT, config.CONV1D_2_OUTPUT], stddev=0.1, dtype=tf.float32), trainable= trainable,name="conv1d_kernel_2",)
            self.weight_1 = tf.Variable(tf.random.truncated_normal(shape=[(self.threshold_k - config.CONV1D_2_FILTER_WIDTH + 1) *config.CONV1D_2_OUTPUT, config.DENSE_NODES], stddev=0.1 ,dtype=tf.float32), trainable= trainable,name="weight_1")
            self.bias_1 = tf.Variable(tf.zeros(shape=[config.DENSE_NODES],dtype=tf.float32), trainable= trainable,name="bias_1")
            self.weight_2 = tf.Variable(tf.random.truncated_normal(shape=[config.DENSE_NODES, config.num_classes],dtype=tf.float32), trainable= trainable,name="weight_2")
            self.bias_2 = tf.Variable(tf.zeros(shape=[config.num_classes],dtype=tf.float32), trainable= trainable,name="bias_2")
            self.opt = tf.optimizers.Adam(learning_rate=config.learning_rate, clipnorm=config.max_gradient_norm)
            self.flatten_layer = tf.keras.layers.Flatten()
            self.dropout_layer = tf.keras.layers.Dropout(config.dropout)
            '''
            #self.flatten_layer = tf.keras.layers.Flatten()
            #self.dropout_layer = tf.keras.layers.Dropout(config.dropout)
            self.first_conv_layer = layers.Conv1D(config.CONV1D_1_OUTPUT, self.dim, strides=self.dim, padding='same', activation='relu')
            self.first_pooling_layer = layers.MaxPooling1D()
            self.second_conv_layer = layers.Conv1D(config.CONV1D_2_OUTPUT, config.CONV1D_2_FILTER_WIDTH, padding='same', activation='relu')
            self.flatten_layer = tf.keras.layers.Flatten()
            self.dense_layer = layers.Dense(config.DENSE_NODES, activation=tf.nn.relu)
            # Dropout, to avoid overfitting
            self.dropout_layer = tf.keras.layers.Dropout(config.dropout)
            # Readout layer
            self.output_layer = layers.Dense(config.num_classes)

    def call(self, D_inverse, A_tilde, X, nodes_size_list, is_train, sparse = False):
        node_size_pl, D_inverse_pl, A_tilde_pl, X_pl, is_train = self.create_input(D_inverse, A_tilde, X, nodes_size_list, is_train, sparse)
        return self.predict(node_size_pl, D_inverse_pl, A_tilde_pl, X_pl, is_train, sparse)

    def create_input(self, D_inverse, A_tilde, X, nodes_size_list, is_train, sparse = False):
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
            D_inverse_pl = tf.cast(D_inverse.todense(), dtype=tf.float32)
            A_tilde_pl = tf.cast(A_tilde.todense(), dtype=tf.float32)
            X_pl = tf.cast(X, dtype=tf.float32)
            node_size_pl = nodes_size_list
        #X_pl=self.normalization_layer(X_pl)    
        return node_size_pl, D_inverse_pl, A_tilde_pl, X_pl, is_train

    def predict(self, node_size_pl, D_inverse_pl, A_tilde_pl, X_pl, is_train, sparse):
        is_train = is_train
        # GRAPH CONVOLUTION LAYER
        '''
        gl_1_XxW = dot(X_pl, self.graph_weight_1, sparse)
        gl_1_AxXxW = dot(A_tilde_pl, gl_1_XxW, sparse)
        Z_1 = tf.nn.tanh(gl_1_AxXxW)
        gl_2_XxW = tf.matmul(Z_1, self.graph_weight_2)
        gl_2_AxXxW = dot(A_tilde_pl, gl_2_XxW, sparse)
        Z_2 = tf.nn.tanh(gl_2_AxXxW)
        gl_3_XxW = tf.matmul(Z_2, self.graph_weight_3)
        gl_3_AxXxW = dot(A_tilde_pl, gl_3_XxW, sparse)
        Z_3 = tf.nn.tanh(gl_3_AxXxW)
        gl_4_XxW = tf.matmul(Z_3, self.graph_weight_4)
        gl_4_AxXxW = dot(A_tilde_pl, gl_4_XxW, sparse)
        Z_4 = tf.nn.tanh(gl_4_AxXxW)
        '''
        Z_1 = self.hidden_layer1(X_pl, A_tilde_pl, False, sparse)
        Z_2 = self.hidden_layer2(Z_1, A_tilde_pl, False, sparse)
        Z_3 = self.hidden_layer3(Z_2, A_tilde_pl, False, sparse)
        Z_4 = self.hidden_layer4(Z_3, A_tilde_pl, False, sparse)
        graph_conv_output = tf.concat([Z_1, Z_2, Z_3, Z_4], axis=-1)  # shape=(node_size/None, 32+32+32)

        graph_conv_output_stored = [tf.gather(graph_conv_output[i], tf.nn.top_k(Z_4[i][:, -1], config.max_nodes).indices) for i in range(len(node_size_pl))]

        graph_conv_output_top_k = [tf.cond(tf.less(node_size_pl[i], self.threshold_k),
                                        lambda: tf.concat(axis=0,
                                                            values=[graph_conv_output_stored[i][:node_size_pl[i],:],
                                                                    tf.zeros(dtype=tf.float32,
                                                                            shape=[self.threshold_k-node_size_pl[i],self.dim])]),
                                        lambda: tf.slice(graph_conv_output_stored[i], begin=[0, 0], size=[self.threshold_k, -1])) for i in range(len(node_size_pl))]
        
        #graph_conv_output_stored = [tf.gather(graph_conv_output[i], tf.nn.top_k(Z_4[i][:, 0], config.max_nodes).indices) for i in range(len(node_size_pl))]
        
        # FLATTEN LAYER
        emmbed = tf.reshape(graph_conv_output_top_k, shape=[len(node_size_pl), self.dim*self.threshold_k, 1])
        cnn_1d = self.cnn1d_layers(emmbed)
        output = self.fc_layer(cnn_1d)
        #dropout_layer = self.dropout_layer(output, training = is_train)
        pre_y = self.output_layer(output)
        pos_score = tf.nn.softmax(pre_y)
        return pos_score, pre_y, np.argmax(pre_y,-1)
        #print(graph_conv_output_flatten.shape)
        #print(node_size_pl, self.threshold_k, self.dim)
        #assert graph_conv_output_flatten.shape == [len(node_size_pl), self.dim*self.threshold_k, 1]

        # 1-D CONVOLUTION LAYER 1:
        # kernel = (filter_width, in_channel, out_channel)
        #conv_1d_a = tf.nn.conv1d(graph_conv_output_flatten, self.conv1d_kernel_1, stride=self.dim, padding="VALID")
        #print(conv_1d_a.shape)
        #assert conv_1d_a.shape == [len(node_size_pl), self.threshold_k, config.CONV1D_1_OUTPUT]
        # 1-D CONVOLUTION LAYER 2:
        #conv_1d_b = tf.nn.conv1d(conv_1d_a, self.conv1d_kernel_2, stride=1, padding="VALID")
        #print(conv_1d_b.shape)
        #assert conv_1d_b.shape == [len(node_size_pl), self.threshold_k - config.CONV1D_2_FILTER_WIDTH + 1, config.CONV1D_2_OUTPUT]
        #conv_output_flatten = self.flatten_layer(conv_1d_b)

        # DENSE LAYER
        #dense_z = tf.nn.relu(tf.matmul(conv_output_flatten, self.weight_1) + self.bias_1)
        #if tf.math.equal(is_train,True):
            #print('dropout...')
        #    dense_z = self.dropout_layer(dense_z)
        #pre_y = tf.matmul(dense_z, self.weight_2) + self.bias_2
        #pos_score = tf.nn.softmax(pre_y)
        #return pos_score, pre_y, np.argmax(pre_y,-1)

    def cnn1d_layers(self, inputs):
        graph_embeddings = tf.reshape(inputs, [-1, self.threshold_k * self.dim, 1])  # (batch, width, channel)
        first_conv = self.first_conv_layer(graph_embeddings)
        print(first_conv.shape)
        first_conv_pool = self.first_pooling_layer(first_conv)
        print(first_conv_pool.shape)

        second_conv = self.second_conv_layer(first_conv_pool)
        return second_conv

    def fc_layer(self, inputs):
        cnn1d_embed = self.flatten_layer(inputs)
        outputs = self.dense_layer(cnn1d_embed)
        return outputs

    def create_adversarial_pattern(self, D_inverse, A_tilde, X, nodes_size_list, label, is_train=False, sparse = True):
        node_size_pl, D_inverse_pl, A_tilde_pl, X_pl, is_train = self.create_input( D_inverse, A_tilde, X, [nodes_size_list], False, True)
        D_inverse_pl = [tf.sparse.to_dense(d) for d in D_inverse_pl]
        A_tilde_pl = [tf.sparse.to_dense(d) for d in A_tilde_pl]
        X_pl = [tf.sparse.to_dense(d) for d in X_pl]
        with tf.GradientTape() as tape:
            tape.watch(X_pl)
            pos_score, pre_y, new_res =self.predict(node_size_pl, D_inverse_pl, A_tilde_pl, X_pl, False, False)
            loss = self.cal_loss(label, pre_y)
            
        gradient = tape.gradient(loss, X_pl)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        return signed_grad, new_res

    def compute_node_gradients(self, D_inverse_pl, A_tilde_pl, X_pl, node_size_pl, class_of_interest, is_train=False, sparse=False):
        with tf.GradientTape() as tape:
            tape.watch(X_pl)
            _, logits, new_res = self.predict(node_size_pl, D_inverse_pl, A_tilde_pl, X_pl, False, False)
            output = tf.nn.softmax(logits)
            cost_value = K.gather(output[0], class_of_interest)
        node_gradients = tape.gradient(cost_value, X_pl)
        return node_gradients, new_res
  
    def create_adversarial_pattern2(self, D_inverse, A_tilde, X, nodes_size_list, class_of_interest, is_train=False, sparse=True,steps=10):
        node_size_pl, D_inverse_pl, A_tilde_pl, X_pl, is_train = self.create_input( D_inverse, A_tilde, X, [nodes_size_list], False, True)
        D_inverse_pl = [tf.sparse.to_dense(d) for d in D_inverse_pl]
        A_tilde_pl = [tf.sparse.to_dense(d) for d in A_tilde_pl]
        X_pl = [tf.sparse.to_dense(d) for d in X_pl]

        X_val = tf.convert_to_tensor(X_pl)
        A_val = tf.convert_to_tensor(A_tilde_pl)
        total_gradients = tf.convert_to_tensor(np.zeros(X_val.shape), dtype=tf.float32)
        X_baseline = tf.convert_to_tensor(np.zeros(X_val.shape), dtype=tf.float32)
        X_diff = tf.convert_to_tensor(X_val  - X_baseline, dtype=tf.float32)
        for alpha in np.linspace(0., 1., steps):
            X_step = X_baseline + alpha * X_diff
            grad, new_res= self.compute_node_gradients(D_inverse_pl, A_val, X_step, node_size_pl, class_of_interest)
            total_gradients += grad
        #gradients = np.squeeze(total_gradients * X_diff, 0)
        signed_grad = tf.sign(total_gradients)
        return signed_grad, new_res

    def cal_loss(self,label, logits):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)) 


import pdb
def train():
    #with open(config.data_lens, 'rb') as jf:
    #    train_graph_paths, val_graph_paths, test_graph_paths, lbs_train, lbs_val, lbs_test  = pickle.load(jf)
    #_, ckpt_manager = load_ckpt(model, 'models/gnn-mc/'+config.current_malware)
    with open(config.data_lens, 'rb') as jf:
        train_graph_paths, val_graph_paths, test_graph_paths, lbs_train, lbs_val, lbs_test  = pickle.load(jf)
    
    config.learning_rate = 0.00100
    for i in range(6):
        #train_graph_paths, test_set = paths[train_index], paths[test_index]
        #lbs_train, lbs_test = lbs[train_index], lbs[test_index]
        #val_graph_paths, test_graph_paths, lbs_val, lbs_test = train_test_split(test_set, lbs_test, test_size=0.7)
        #k = (train_graph_paths, val_graph_paths, test_graph_paths, lbs_train, lbs_val, lbs_test)
        with open(config.data_lens+'-gcn-'+str(i)+'.txt', 'rb') as jf:
            train_graph_paths, val_graph_paths, lbs_train, lbs_val  = pickle.load(jf)

        dics = read_all_data_proc(rewrite=False)#,path="malware/"+config.current_malware+config.current_feats+"gnn/all_data_adj_feature")

        model_name='models/gnn-mc-2-7/'+str(i)+'/'+config.current_malware#+'add-ri-gene-30/'
        #model_name='models-1/gnn-mc/'+config.current_malware
        #config.top_k = 1000
        model = GNN_Detector(config.vector_size, config.top_k)
        _, ckpt_manager = load_ckpt(model, model_name)#, 'models/gnn-mc/'+config.current_malware+'ckpt-20')
        print(model_name)

        '''
        if config.add_ri_gene:
            a = []#list(range(len(config.insert_dead_opcode_seq)))
            a.extend(['all'])
            print(a)
            for _idx in a:
                for attack in ['sri','sai','sgi','srl']:
                    save_path = "malware/"+config.current_malware+'generated_ri/'+attack+'/malware_30/gnn/'+str(_idx)+"_data_adj_feature"                  
                    with open(save_path, "rb") as f_out:
                            new_dics = pickle.load(f_out)
                            if len(list(new_dics.keys())) > 500:
                                new_paths = random.choices(list(new_dics.keys()), k = 500)
                            else:
                                new_paths = list(new_dics.keys())
                            for path in new_paths:
                                train_graph_paths=np.append(train_graph_paths,str(_idx)+attack+path)
                                dics[str(_idx)+attack+path]=new_dics[path]
        '''
        print(len(train_graph_paths))
        #'''
        #'''
        #exit()
        #exit()
        train_detector_batch(model, ckpt_manager, 'gnn', dics, train_graph_paths, test_graph_paths, val_graph_paths)
        test_detector(model, ckpt_manager, 'gnn', dics, train_graph_paths, train_graph_paths,0)
        test_detector(model, ckpt_manager, 'gnn', dics, train_graph_paths, test_graph_paths, 0)
        test_detector(model, ckpt_manager, 'gnn', dics, train_graph_paths, val_graph_paths, 0)
        #show_node_importance(model, ckpt_manager, 'gnn', dics, train_graph_paths, test_graph_paths, 0)

    

def compare():
    with open(config.data_lens, 'rb') as jf:
        train_graph_paths, val_graph_paths, test_graph_paths, lbs_train, lbs_val, lbs_test  = pickle.load(jf)
    #_, ckpt_manager = load_ckpt(model, 'models/gnn-mc/'+config.current_malware)
    
    dics = read_all_data_proc(rewrite=False)

    model_name='models/gnn-mc/'+config.current_malware+'add-ri-gene-30/'
    #config.top_k = 1000

    model = GNN_Detector(config.vector_size, config.top_k)
    _, ckpt_manager = load_ckpt(model, model_name)#, 'models/gnn-mc/final-3-all-btbv/ckpt-20')
    print(model_name)
    d2 = test_detector(model, ckpt_manager, 'gnn', dics, train_graph_paths, test_graph_paths, 0)


    model = GNN_Detector(config.vector_size, config.top_k)
    _, ckpt_manager = load_ckpt(model, model_name)
    print(model_name)
    d1 = test_detector(model, ckpt_manager, 'gnn', dics, train_graph_paths, test_graph_paths, 0)

    savefig([d2,d1],10086)
    
if __name__ == '__main__':
    train()
    #compare()
