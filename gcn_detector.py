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
import random
import json
from utils import *
from layers import ConvolutionalLayer, PoolingLayer
from sklearn.model_selection import KFold
class GCN_Detector(tf.keras.Model):
    def __init__(self, trainable, featureless=False, debug=True):
        super(GCN_Detector, self).__init__(name='')
        self.output_dim = config.num_classes
        self.input_dim = config.vector_size
        self.featureless = featureless
        self.init_model(trainable)

    def init_model(self, trainable):
        #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(config.learning_rate,decay_steps=100000,decay_rate=0.96,staircase=True)
        self.opt = tf.optimizers.Adam(learning_rate=config.learning_rate, clipnorm=config.max_gradient_norm)

        with tf.name_scope("GCN_Detector_Model") as scope:
            self.hidden_layer1 = ConvolutionalLayer(input_dim=self.input_dim,
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
            self.pooling_layer = PoolingLayer(input_dim=config.GRAPH_CONV_LAYER_CHANNEL,
                                                    output_dim=self.output_dim,
                                                    activation=tf.nn.leaky_relu,
                                                    sparse_inputs=False,
                                                    featureless = self.featureless,
                                                    trainable = trainable)

    def create_input(self, support, feats, is_train, sparse):    
        if sparse:
            if isinstance(support, list) == False:
                support = [support]
                feats = [feats]
            support = [tf.SparseTensor(tf.cast(_inverse[0],tf.int64),tf.cast(_inverse[1],tf.float32),_inverse[2]) for _inverse in support]
            feats = [tf.SparseTensor(tf.cast(_inverse[0],tf.int64),tf.cast(_inverse[1],tf.float32),_inverse[2]) for _inverse in feats]
        else:    
            support = [tf.cast(support.todense(), dtype=tf.float32)]
            feats = [tf.cast(feats, dtype=tf.float32)]
        return support, feats    

    def predict(self, support, feats, is_train, sparse):
        hidden1 = self.hidden_layer1(feats, support, is_train, sparse)
        hidden2 = self.hidden_layer2(hidden1, support, is_train, sparse)
        hidden3 = self.hidden_layer3(hidden2, support, is_train, sparse)
        embeddings = self.hidden_layer4(hidden3, support, is_train, sparse)

        logits = self.pooling_layer(embeddings, support, is_train, sparse)
        pos_score = tf.nn.softmax(logits)
        return pos_score, logits, np.argmax(logits,1)

    def call(self, support, feats, is_train, sparse):
        support, feats = self.create_input(support, feats, is_train, sparse)
        return self.predict(support, feats, is_train, sparse)

 
    def create_adversarial_pattern(self, support, feats, label, is_train=False, sparse=True):
        support, feats = self.create_input(support, feats, is_train, sparse)
        support = [tf.sparse.to_dense(d) for d in support]
        feats = [tf.sparse.to_dense(d) for d in feats]
        with tf.GradientTape() as tape:
            tape.watch(feats)
            pos_score, pre_y, new_res =self.predict(support, feats, False, False)
            loss = self.cal_loss(label, pre_y)     
        gradient = tape.gradient(loss, feats)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        return signed_grad,new_res
    
    def compute_node_gradients(self, support, feats, class_of_interest, is_train=False, sparse=False):
        with tf.GradientTape() as tape:
            tape.watch(feats)
            _, logits, new_res = self.predict(support, feats, False, False)
            output = tf.nn.softmax(logits)
            cost_value = K.gather(output[0], class_of_interest)
            
        node_gradients = tape.gradient(cost_value, feats)
        return node_gradients, new_res
  
    def create_adversarial_pattern2(self, support, feats, class_of_interest, is_train=False, sparse=True,steps=20):
        support, feats = self.create_input(support, feats, is_train, sparse)
        support = [tf.sparse.to_dense(d) for d in support]
        feats = [tf.sparse.to_dense(d) for d in feats]

        X_val = tf.convert_to_tensor(feats)
        A_val = tf.convert_to_tensor(support)
        total_gradients = tf.convert_to_tensor(np.zeros(X_val.shape), dtype=tf.float32)
        X_baseline = tf.convert_to_tensor(np.zeros(X_val.shape), dtype=tf.float32)
        X_diff = tf.convert_to_tensor(X_val  - X_baseline, dtype=tf.float32)
        for alpha in np.linspace(0., 1., steps):
            X_step = X_baseline + alpha * X_diff
            grad, new_res= self.compute_node_gradients(A_val, X_step, class_of_interest)
            total_gradients += grad
        gradients = np.squeeze(total_gradients * X_diff, 0)
        signed_grad = tf.sign(gradients)
        return signed_grad, new_res

    def cal_loss(self, label, logits):
        #print(label, logits)
        return  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits))

def train():
    with open(config.data_lens+'-all.txt', 'rb') as jf:
        train_graph_paths, val_graph_paths, test_graph_paths, lbs_train, lbs_val, lbs_test  = pickle.load(jf)
    '''
    remove = random.sample(list(range(len(test_graph_paths))),80)
    test_id = list(range(len(test_graph_paths)))
    for i in remove:
        test_id.remove(i)
    val_graph_paths = np.array(val_graph_paths.tolist() + test_graph_paths[remove].tolist())
    lbs_val = np.array(lbs_val.tolist() + lbs_test[remove].tolist())
    k = (train_graph_paths, val_graph_paths, test_graph_paths[test_id], lbs_train, lbs_val, lbs_test[test_id])
    with open(config.data_lens+'-all.txt', 'wb') as jf:
        pickle.dump(k, jf)
    '''
    '''
    #paths = np.array(train_graph_paths.tolist() + val_graph_paths.tolist())# + test_graph_paths.tolist())
    #lbs = np.array(lbs_train.tolist() + lbs_val.tolist())# + lbs_test.tolist())
    idx = list(range(len(train_graph_paths)))
    random.shuffle(idx)
    for i in range(6):
        new_val_path = train_graph_paths[idx[i*800:(i+1)*800]]
        new_val_lbs = lbs_train[idx[i*800:(i+1)*800]]
        new_train_path = np.array(train_graph_paths[idx[0:i*800]].tolist() +train_graph_paths[idx[(i+1)*800:]].tolist() + val_graph_paths.tolist())
        new_train_lbs = np.array(lbs_train[idx[0:i*800]].tolist() +lbs_train[idx[(i+1)*800:]].tolist() + lbs_val.tolist())
        k = (new_train_path, new_val_path, new_train_lbs, new_val_lbs)
        print(len(new_train_path),len(new_val_path),len(new_train_lbs),len(new_val_lbs))
        print(new_train_path[:5], new_val_path[:5], new_train_lbs[:5], new_val_lbs[:5])
        print([j for j in new_train_path if j in new_val_path])
        print([j for j in new_train_path if j in test_graph_paths])
        print([j for j in new_val_path if j in test_graph_paths])
        with open(config.data_lens+'-2gcn-'+str(i)+'.txt', 'wb') as jf:
            pickle.dump(k, jf)
    exit()
    '''
    config.learning_rate = 0.00100
    for i in range(6):
        
        with open(config.data_lens+'-gcn-'+str(i)+'.txt', 'rb') as jf:
            train_graph_paths, val_graph_paths, lbs_train, lbs_val  = pickle.load(jf)

        dics = read_all_data_proc(rewrite=False)
        model_name='models/gcn-mc-tanh-7-cross/'+str(i)+'/'+config.current_malware#+'add-ri-gene-30/'
        model = GCN_Detector(trainable = True)
        _, ckpt_manager = load_ckpt(model, model_name)#, 'models/gcn-mc/'+config.current_malware+'ckpt-27')
        print(model_name)
        '''
        if config.add_ri_gene:
            a = []#list(range(len(config.insert_dead_opcode_seq)))
            a.extend(['all'])
            print(a)
            for _idx in a:
                for attack in ['sri','sai','sgi','srl']:
                    save_path = "malware/"+config.current_malware+'generated_ri/'+attack+'/malware_30/gcn/'+str(_idx)+"_data_adj_feature"                  
                    with open(save_path, "rb") as f_out:
                            new_dics = pickle.load(f_out)
                            if len(list(new_dics.keys())) > 500:
                                new_paths = random.choices(list(new_dics.keys()), k = 500)
                            else:
                                new_paths = list(new_dics.keys())
                            for path in new_paths:
                                #print(str(_idx)+attack+path)
                                train_graph_paths=np.append(train_graph_paths,str(_idx)+attack+path)
                                dics[str(_idx)+attack+path]=new_dics[path]
        
        print(len(train_graph_paths),len(dics))
        #exit()
        '''
        train_detector_batch(model, ckpt_manager, 'gcn', dics, train_graph_paths, test_graph_paths, val_graph_paths)
        test_detector(model, ckpt_manager, 'gcn', dics, train_graph_paths, train_graph_paths,0)
        test_detector(model, ckpt_manager, 'gcn', dics, train_graph_paths, test_graph_paths, 0)
        test_detector(model, ckpt_manager, 'gcn', dics, train_graph_paths, val_graph_paths, 0)

def compare():
    with open(config.data_lens, 'rb') as jf:
        train_graph_paths, val_graph_paths, test_graph_paths, lbs_train, lbs_val, lbs_test  = pickle.load(jf)
    
    dics = read_all_data_proc(rewrite=False)

    model_name='models/gcn-mc/'+config.current_malware+'add-ri-gene-30/'
    #config.top_k = 1000

    model = GCN_Detector(trainable = False)
    _, ckpt_manager = load_ckpt(model, 'models/gcn-mc/'+config.current_malware)
    print(model_name)
    d2 = test_detector(model, ckpt_manager, 'gcn', dics, train_graph_paths, test_graph_paths, 0)
    
    model = GCN_Detector(trainable = False)
    _, ckpt_manager = load_ckpt(model, model_name)
    print(model_name)
    d1 = test_detector(model, ckpt_manager, 'gcn', dics, train_graph_paths, test_graph_paths, 0)

    savefig([d2,d1],10086)
if __name__ == '__main__':
    train()
    #compare()

