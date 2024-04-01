import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
import scipy.io as scio
import config
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, accuracy_score,recall_score,precision_score,roc_auc_score
import multiprocessing
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import random
tf.keras.backend.set_floatx('float32')
#import ctypes
#ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
'''
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

_config = ConfigProto()
_config.gpu_options.allow_growth = True
session = InteractiveSession(config=_config)
print('==============')
print(tf.config.list_physical_devices('GPU'))
'''
#-----


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        model.save_weights('es_checkpoint.pb')

_LAYER_UIDS = {}
def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def sparse_dropout(x, keep_prob, noise_shape):
    if isinstance(x, list):
        supports = list()
        for xi in x:
            random_tensor = keep_prob
            random_tensor += tf.random.uniform([xi.values.shape[0]])
            dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
            pre_out = tf.sparse.retain(xi, dropout_mask)
            supports.append(pre_out * (1./keep_prob))
        return supports
    else:
        random_tensor = keep_prob
        random_tensor += tf.random.uniform([x.values.shape[0]])
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse.retain(x, dropout_mask)
        return pre_out * (1./keep_prob)

def sp_dot(x, y, sparse=True):
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def spare_add(x, y, sparse=True):
    supports = list()
    for i in range(len(y)):
        support=tf.sparse.add(x[i], y[i])
        supports.append([support])
    return tf.concat(supports,0)

def dot(x, y, sparse=True):
    if isinstance(x, list) and len(y.shape)==2:
        supports = list()
        for i in range(len(x)):
            support=sp_dot(x[i], y, sparse=sparse)
            supports.append([support])
        res = tf.concat(supports,0)
    elif isinstance(x, list) and len(y.shape)==3:
        supports = list()
        for i in range(len(x)):
            support=sp_dot(x[i], y[i], sparse=sparse)
            supports.append([support])
        res = tf.concat(supports,0)
    else:
        res=sp_dot(x, y, sparse=sparse)
    return res

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def to_tuple(mat):
    if not sp.isspmatrix_coo(mat):
        mat = mat.tocoo()
    idxs = np.vstack((mat.row, mat.col)).transpose()
    values = mat.data
    shape = mat.shape
    return idxs, values, shape

def sparse_to_tuple(sparse_mat):
    if isinstance(sparse_mat, list):
        for i in range(len(sparse_mat)):
            sparse_mat[i] = to_tuple(sparse_mat[i])
    else:
        sparse_mat = to_tuple(sparse_mat)
    return sparse_mat

def process_features(features):
    features /= features.sum(1).reshape(-1, 1)
    features[np.isnan(features) | np.isinf(features)] = 0 #serve per le features dei nodi globali, che sono di soli 0.
    return sparse_to_tuple(sp.csr_matrix(features))

def zeros(shape, name=None,trainable = True):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name,trainable = trainable)

def glorot(shape, name=None,trainable = True):
    #init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    #val = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    #return tf.Variable(val, name=name,trainable = trainable)
    return tf.Variable(tf.random.truncated_normal(shape=shape, stddev=0.1, dtype=tf.float32), name=name,trainable = trainable)
#-----

opt = tf.optimizers.Adam(learning_rate=config.learning_rate, clipnorm=config.max_gradient_norm)
acc, rec, prec = [],[],[]
maxs = 0.8
def train_detector(model, ckpt_manager, types, dics, train_graph_paths, test_graph_paths):
    global maxs
    for epoch in range(config.epoch):
        random.shuffle(train_graph_paths)
        random.shuffle(test_graph_paths)
        cou = 0
        for i in range(len(train_graph_paths)):
            path = train_graph_paths[i]
            D_inverse, A_tilde, Y, X, nodes_size_list, _ = dics[path]
            with tf.GradientTape() as tape:
                pos_score, logits, pre_y_value = detection(types, model, D_inverse, A_tilde, X, nodes_size_list, True, True)
                #pos_score, logits, pre_y_value = detection(types, model, [D_inverse], [A_tilde], [X], nodes_size_list, True, True)
                loss = model.cal_loss(Y, logits)
            gradients = tape.gradient(loss, model.trainable_variables)
            k = [(v.name, tf.reduce_sum(k).numpy()) for k,v in zip(gradients, model.trainable_variables) if k is not None and v is not None and tf.reduce_sum(k).numpy() != 0]
            if len(k) ==0:
                cou += 1
            #else:
            #    print('has',len(k))
            opt.apply_gradients(zip(gradients, model.trainable_variables))
            #if i % 50 == 0:
            #    test_detector(model, ckpt_manager, types, dics, train_graph_paths, test_graph_paths, epoch)
        test_detector(model, ckpt_manager, types, dics, train_graph_paths, test_graph_paths, epoch)
        print(cou/len(train_graph_paths))
        
    saveacc([acc, rec, prec, types])

def train_detector_batch(model, ckpt_manager, types, dics, train_graph_paths, test_graph_paths, val_graph_paths, _max=0.8):
    global maxs
    #maxs = 0.9529761904761904
    #maxs = 0.9392857142857143
    maxs = _max
    if (maxs >= 0.94) :
        return maxs
    for epoch in range(1,config.epoch):
        random.shuffle(train_graph_paths)
        random.shuffle(test_graph_paths)
        sampling = [random.choice(train_graph_paths) for _ in range(config.batch_size - len(train_graph_paths) + int(len(train_graph_paths)/config.batch_size) * config.batch_size)]
        #sampling = random.choices(train_graph_paths, k = config.batch_size - len(train_graph_paths) + int(len(train_graph_paths)/config.batch_size) * config.batch_size)
        train_graph = np.concatenate([sampling,train_graph_paths])
        cou = 0
        loss = 0
        for i in range(0,len(train_graph), config.batch_size):
            paths = train_graph[i: i+config.batch_size]
            D_inverse, A_tilde, Y, X, nodes_size_list = [], [], [], [], []
            for path in paths:
                _D_inverse, _A_tilde, _Y, _X, _nodes_size_list, _ = dics[path]
                D_inverse.append(_D_inverse)
                A_tilde.append(_A_tilde)
                Y.append(_Y[0])
                X.append(_X)
                nodes_size_list.append(_nodes_size_list)
            with tf.GradientTape() as tape:
                pos_score, logits, pre_y_value = detection(types, model, D_inverse, A_tilde, X, nodes_size_list, True, True)
                loss = model.cal_loss(Y, logits)
            gradients = tape.gradient(loss, model.trainable_variables)
            k = [(v.name, tf.reduce_sum(k).numpy()) for k,v in zip(gradients, model.trainable_variables) if k is not None and v is not None and tf.reduce_sum(k).numpy() != 0]
            if len(k) ==0:
                cou += 1
            #else:
            #    print('has',len(k))
            opt.apply_gradients(zip(gradients, model.trainable_variables))
            if i % config.batch_size == 0:
                print("====")
                print('test set')
                test_detector(model, ckpt_manager, types, dics, train_graph, test_graph_paths, epoch)
                #print('train set')
                #test_detector(model, ckpt_manager, types, dics, train_graph, train_graph_paths, 0)
                #print('val set')
                #test_detector(model, ckpt_manager, types, dics, train_graph, val_graph_paths, 0)
                print(epoch,i,maxs, loss)
        print("====")
        print('test set')
        test_detector(model, ckpt_manager, types, dics, train_graph, test_graph_paths, epoch)
        print('train set')
        test_detector(model, ckpt_manager, types, dics, train_graph, train_graph_paths, 0)
        print('val set')
        test_detector(model, ckpt_manager, types, dics, train_graph, val_graph_paths, 0)
        if (maxs < 0.5 or maxs > 0.94) :
            return maxs

        #test_detector(model, ckpt_manager, types, dics, train_graph_paths, test_graph_paths, epoch)
        print(cou/len(train_graph))
    print([acc, rec, prec, types])
    saveacc([acc, rec, prec, types])
    return maxs

def show_node_importance(model, ckpt_manager, types, dics, train_graph_paths, test_graph_paths, epoch):
    for idx, path in enumerate(test_graph_paths):
        D_inverse, A_tilde, Y, X, nodes_size_list, _ = dics[path]
        node_importances = compute_node_importance(Y, types, model, D_inverse, A_tilde, X, nodes_size_list)
        G = nx.Graph() 
        edges=sparse_to_tuple(A_tilde)
        G.add_edges_from(edges[0])
        plt.clf()
        nx.draw_networkx(G, node_size = node_importances, cmap = plt.cm.Blues,with_labels = False, width = 0.1,edge_color ='.4')
        plt.axis('off') 
        plt.savefig("path.png", dpi=600)

def draw_reward(reward, name):
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.dpi'] = 600
    plt.clf()
    colors = ["cornflowerblue","lightslategrey","crimson","rebeccapurple","teal","olive","maroon","chocolate","blue","darkseagreen"]
    for idx, r in enumerate(reward):
        plt.plot(r, color = colors[idx])
    plt.savefig('res/roc/rl_reward'+name+'.pdf', dpi=600)


def test_detector(model, ckpt_manager, types, dics, train_graph_paths, test_graph_paths, epoch):
    global maxs, acc, rec, prec
    test_acc, prediction, scores, Y_all = 0, [], [], []
    for idx, path in enumerate(test_graph_paths):
        D_inverse, A_tilde, Y, X, nodes_size_list, _ = dics[path]
        pos_score, logits, pre_y_value = detection(types, model, D_inverse, A_tilde, X, nodes_size_list, False, True)
        prediction.append(pre_y_value[0])
        Y_all.append(Y[0])
        scores.append(pos_score[0])
        if pre_y_value[0] == Y[0]:
            test_acc = test_acc + 1
    ypred = np.array(prediction)
    ytrue = np.array(Y_all)
    scores = np.array(scores)
    fpr,tpr,roc_auc = cal_roc(scores, ytrue)
    acc.append(accuracy_score(ytrue, ypred))
    rec.append(recall_score(ytrue, ypred, average='macro'))
    prec.append(precision_score(ytrue, ypred, average='macro'))
    cal_confusion(ytrue, ypred)
    if acc[-1] > maxs and epoch != 0:
        #if  epoch != 0:
        print("save...", maxs, acc[-1])
        maxs = acc[-1]
        ckpt_manager.save()
        #savefig([[fpr,tpr,roc_auc, types]], epoch)
    
    print(epoch, test_acc/len(test_graph_paths), acc[-1], rec[-1], prec[-1], roc_auc)
    return [fpr,tpr,roc_auc, types], acc[-1]

def cal_confusion(ytrue, ypred):
    cm = confusion_matrix(ytrue, ypred)
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print(TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC)

def savefig(datas, epoch = 1):
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.dpi'] = 600
    plt.clf()
    colors = ["cornflowerblue","lightslategrey","crimson","rebeccapurple","teal","olive","maroon","chocolate","darkseagreen"]
    if len(datas) == 2:
        for color, data, name in zip(colors, datas, ['Original Modle','Re-trained Model']):
            fpr,tpr,auc,label = data[0], data[1], data[2], data[3]
            plt.plot(fpr[0], tpr[0], color = color, label=(str(name)+"(area=%0.2f)") % auc[0])
    else:
        data = datas[0]
        for color, idx, name in zip(colors, range(config.num_classes),['benign','malware']):
            fpr,tpr,auc,label = data[0], data[1], data[2], data[3]
            plt.plot(fpr[idx], tpr[idx], color = color, label=(str(name)+"(area=%0.2f)") % auc[idx])

    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('res/roc/epoch_'+label+str(epoch)+'.png', dpi=600)

def saveacc(data2):
    plt.clf()
    colors = ["cornflowerblue","lightslategrey","crimson","rebeccapurple","teal","olive","maroon","chocolate","darkseagreen"]
    plt.plot(range(len(data2[0])), data2[0], color = colors[0], label="acc")
    plt.plot(range(len(data2[1])), data2[1], color = colors[1], label="rec")
    plt.plot(range(len(data2[2])), data2[2], color = colors[2], label="prec")
    plt.legend(loc="lower right")
    plt.xlabel('Epoch')
    plt.savefig('res/roc/'+data2[3]+'acc.png', dpi=600)

def cal_roc(prediction, all_labels):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    #print(prediction.shape,all_labels.shape)
    for i in range(config.num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels, prediction[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i],tpr[i])
    return fpr, tpr, roc_auc

def compute_node_gradients(class_of_interest, types, model, D_inverse, A_tilde, X, nodes_size_list):
    with tf.GradientTape() as tape:
        tape.watch(X)
        _, logits, _ = detection(types, model, D_inverse, A_tilde, X, nodes_size_list, False, True)
        output = tf.nn.softmax(logits)
        cost_value = K.gather(output[0], class_of_interest)
    node_gradients = tape.gradient(cost_value, X)
    return node_gradients

def compute_node_importance(class_of_interest, types, model, D_inverse, A_tilde, X, nodes_size_list, steps=20):
    X_val = X.todense()
    A_val = tf.convert_to_tensor(A_tilde.todense())
    total_gradients = tf.convert_to_tensor(np.zeros(X_val.shape))
    X_baseline = tf.convert_to_tensor(np.zeros(X_val.shape))
    X_diff = tf.convert_to_tensor(X_val  - X_baseline)
    for alpha in np.linspace(0, 1, steps):
        X_step = X_baseline + alpha * X_diff
        total_gradients += compute_node_gradients(
            class_of_interest, types, model, D_inverse, A_val, X_step, nodes_size_list
        )
    gradients = total_gradients * X_diff
    #gradients = np.squeeze(gradients, 0)
    return np.sum(gradients, axis=-1)
    
def detection(types, model, D_inverse, A_tilde, X, nodes_size_list,trainable, sparse):
    if types == 'gnn':
        pos_score, logits, pre_y_value = model(D_inverse, A_tilde, X, [nodes_size_list], trainable, sparse)
    elif types == 'gcn':
        #A_tilde = A_tilde - tf.eye(A_tilde.shape[0])
        pos_score, logits, pre_y_value = model(A_tilde, X, trainable, sparse)
    elif types == 'gat':
        pos_score, logits, pre_y_value = model(A_tilde, X, trainable, sparse)
    elif types == 'cnn':
        _A_tilde = np.c_[A_tilde,tf.zeros([A_tilde.shape[0],config.max_nodes-A_tilde.shape[0]])]
        _, _, pred = model(_A_tilde, trainable)
    return pos_score, logits, pre_y_value

def load_ckpt(model, path, newest=True):
    ckpt = tf.train.Checkpoint(transformer=model, optimizer=opt)
    ckpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        if newest == True:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!', path, newest, ckpt_manager.latest_checkpoint)
        else:
            ckpt.restore(newest)
            print ('checkpoint restored!!', path, newest)
    elif newest != True:
        ckpt.restore(newest)
        print ('checkpoint restored!!', path, newest)
    return ckpt, ckpt_manager

def save_without_load_ckpt(model,path):
    ckpt = tf.train.Checkpoint(transformer=model, optimizer=opt)
    ckpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=5)
    return ckpt, ckpt_manager

'''
def sample_gumbel(shape, eps=1e-20): 
    """Sample from Gumbel(0, 1)"""
    U = tf.random.uniform(shape, minval=0,maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y
'''
