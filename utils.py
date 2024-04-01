import glob
from operator import itemgetter
import pickle
#import queue
import threading
from tf_utils import *
import config
import os
import json
import os.path
import glob
from operator import itemgetter
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from scipy import sparse

MAX=32

import hashlib
all_files = []
def scan_file(files, paths):     
  for localpath in paths:                              
    for _fn in os.listdir(localpath):                                       
      fn = os.path.join(localpath, _fn)            
      if not os.path.isdir(fn):    
        _fn = _fn.lower()                                        
        if _fn not in all_files:# and _fn.find('.dll') == -1:
          all_files.append(_fn)       
          files.append(fn)                                             
      else:                                                                
        scan_file(files, [fn])     

### step 1
def return_label(fn):
  try:
    if fn.find('benign/') != -1:
      return config.labels['ben']
    elif fn.find('malware/Backdoor') != -1:
      return config.labels['back']
    elif fn.find('malware/Trojan') != -1:
      return config.labels['troj']
    elif fn.find('malware/Worm') != -1:
      return config.labels['worm']
    elif fn.find('malware/Ransomware') != -1:
      return config.labels['rans']
    elif fn.find('malware/Virus') != -1:
      return config.labels['virus']
    else:
      print('wrong name', fn)
      exit()
  except:
    return -1

def split_data(rewrite):
  print('split... ', os.path.dirname(config.data_lens))
  if rewrite or os.path.exists(config.data_lens) == False:
    os.makedirs(os.path.dirname(config.data_lens), exist_ok=True)
    if os.path.exists('malware/temp-3') == False:
      files = []
      global all_files
      all_files = []
      scan_file(files, config.data_dir_benign)
      scan_file(files, config.data_dir_malware)
      #print(files, all_files)
      lens = {}
      for fn in files:
        with open(fn, 'r') as jf:
          try:
            strs = json.load(jf)
            #print(len(strs['blocks']))
            lens[fn] = len(strs['blocks'])
          except:
            print(fn)
            pass
      with open('malware/temp-3', 'wb') as jf:
        pickle.dump(lens, jf)
    else:
      with open('malware/temp-3', 'rb') as jf:
        lens = pickle.load(jf)

    all_prog = [[] for _ in config.labels]
    all_prog_len = [[] for _ in config.labels]
    unique = set()
    for st,v in lens.items():
      a = st.split('/')[-1].lower()
      if a in unique:
        print('not unique', st)
        continue
      else:
        unique.add(a)
      if v >= 12 and v <= config.max_nodes:
        label = return_label(st)
        if label != -1:
          all_prog[label].append(st)
          all_prog_len[label].append(v)

    for i in set(config.labels.values()):
      m = pd.DataFrame({i:all_prog_len[i]})
      print(m.describe())
    
    resorted=[list(map(list,zip(*sorted(zip(all_prog_len[i],all_prog[i]))))) for i in range(len(config.labels))]
    #_min = min([len(all_prog[i]) for i in range(len(config.labels))])
    _min = 4000
    for i in set(config.labels.values()):
      all_prog[i] = resorted[i][1][:_min]
      all_prog_len[i] = resorted[i][0][:_min]
    
    for i in set(config.labels.values()):
      m = pd.DataFrame({i:all_prog_len[i]})
      print(m.describe())
    alls = []
    alls.extend(all_prog_len[0])
    alls.extend(all_prog_len[1])
    m = pd.DataFrame({'all':alls})
    print(m.describe())
    exit()

    alls = []
    lbs = []
    for i in range(len(config.labels)):
      alls.extend(all_prog[i])
      lbs.extend([i]*len(all_prog[i]))

    alls = np.array(alls)
    lbs = np.array(lbs)
    train_set, test_set, lbs_train, lbs_test = train_test_split(alls, lbs, test_size=0.3)
    val_set, test_set, lbs_val, lbs_test = train_test_split(test_set, lbs_test, test_size=0.7)
    k = (train_set, val_set, test_set, lbs_train, lbs_val, lbs_test)
    for i in k:
      print(len(i))
    with open(config.data_lens, 'wb') as jf:
      pickle.dump(k, jf)
    return k
  else:
    print('reload used split...')
    with open(config.data_lens, 'rb') as jf:
      k = pickle.load(jf)
      train_set, val_set, test_set, lbs_train, lbs_val, lbs_test = k
      print(len(train_set),len(test_set), len(val_set))
      print(len(lbs_train), np.sum(lbs_train == 0), np.sum(lbs_train == 1), np.sum(lbs_train == 2), np.sum(lbs_train == 3))
      print(len(lbs_test),  np.sum(lbs_test == 0),np.sum(lbs_test == 1),np.sum(lbs_test == 2),np.sum(lbs_test == 3))
      print(len(lbs_val),  np.sum(lbs_val == 0),np.sum(lbs_val == 1),np.sum(lbs_val == 2),np.sum(lbs_val == 3))
      return k


### step 2
def load_opcode_voc(rewrite=True, files=[]):
  if len(files) == 0:
    files = []
    global all_files
    all_files = []
    scan_file(files, config.data_dir_benign)
    scan_file(files, config.data_dir_malware)
  if rewrite or os.path.exists(config.opcode_voc) == False:
    op2 = set([config.split_token])
    mx = [0]
    for fn in files:
      map_opcode(fn, op2, mx)
    opcode_voc = {op:idx for idx, op in enumerate(op2)}
    with open(config.opcode_voc, 'w') as jf:
      json.dump(opcode_voc, jf)
    if config.current_feats != 'opcode/':
      config.vector_size_bytes = mx[0]
      config.vector_size = config.vector_size_bytes
    else:
      config.vector_size = len(opcode_voc)
    print("........................",'\n','Notice: if you want to use bytes, max =',mx[0],'\n',"........................")
    print("........................",'\n','Notice: if you want to use opcode, max =',len(opcode_voc),'\n',"........................")
    return files, opcode_voc
  else:
    if config.current_feats != 'opcode/':
      config.vector_size = config.vector_size_bytes
    print('load used voc...')
    with open(config.opcode_voc, 'r') as jf:
      opcode_voc = json.load(jf)
    return files, opcode_voc

def map_opcode(fn, op2, mx):
  with open(fn, 'r') as jf:
    strs = json.load(jf)
    for idx, block in enumerate(strs['blocks']):
      instructions = block[4]
      if instructions == None:
        continue
      for instruction in instructions.split('\n'):
        opcodes = instruction.split('\t')
        if (len(opcodes)==3):
          op2.add(opcodes[1])
      if block[3] != None:
        mx[0] = max(len(str(bin(int(str(block[3]),16)))), mx[0])

def extract_voc(train_set, val_set, test_set, lbs_train, lbs_val, lbs_test, opcode_rewrite = True):
  print('Loading voc...')
  files = np.concatenate((train_set, test_set, val_set))
  os.makedirs(os.path.dirname(config.opcode_voc), exist_ok=True)
  files, opcode_voc = load_opcode_voc(rewrite=opcode_rewrite, files=files)
  config.vector_size_bytes = len(opcode_voc)
  idx = sorted(opcode_voc.values())
  eye = np.eye(max(idx) + 1)
  features = eye[idx]
  print(features.shape)
  print('Loaded...')
  print(len(train_set),len(test_set))
  print(len(lbs_train), np.sum(lbs_train == 0), np.sum(lbs_train == 1), np.sum(lbs_train == 2), np.sum(lbs_train == 3))
  print(len(lbs_test),  np.sum(lbs_test == 0),np.sum(lbs_test == 1),np.sum(lbs_test == 2),np.sum(lbs_test == 3))
  print(len(lbs_val),  np.sum(lbs_val == 0),np.sum(lbs_val == 1),np.sum(lbs_val == 2),np.sum(lbs_val == 3))
  return opcode_voc, features


### step 3
def retrieve_all_graph(from_path, train_set, val_set, test_set, lbs_train, lbs_val, lbs_test, contruct_rewrite = True):
  with open(from_path, 'rb') as f_in:
     dics = pickle.load(f_in)
  new_dics = {}
  allset = np.concatenate((train_set, test_set, val_set))
  for path in allset:
    D_inverse, adj, Y, X, nodes_size_list, initial_feature_channels = dics[path]
    D_inverse = to_tuple(D_inverse)
    adj = to_tuple(adj)
    X = to_tuple(X)
    Y = [return_label(path)]
    new_dics[path] = (D_inverse, adj, Y, X, nodes_size_list, initial_feature_channels)
  #print(new_dics)
  config.all_data_adj = "malware/"+config.current_malware+config.current_feats+"gnn/all_data_adj_feature"
  os.makedirs(os.path.dirname(config.all_data_adj), exist_ok=True)
  print(config.all_data_adj)
  with open(config.all_data_adj, "wb") as f_out:
      pickle.dump(new_dics, f_out)
  return dics

def new_contruct_all_graph(train_set, val_set, test_set, lbs_train, lbs_val, lbs_test, opcode_voc, features, contruct_rewrite = True):
  allset = np.concatenate((train_set, test_set, val_set))
  #names, all_D_inverse, all_adj, all_Y, all_X, all_nodes_size_list, all_initial_feature_channels = [],[],[],[],[],[],[]
  dics = {}
  os.makedirs(os.path.dirname(config.all_data_adj), exist_ok=True)
  for idx, path in enumerate(allset):
    if (idx%100 ==0):
      print(idx,path)
    data = contruct_graph(path, opcode_voc, features, 1)
    graphs, nodes_size_list, labels = np.array(data['graph']), data['nodes_size'], np.array([data['target']])
    Y = [labels]
    
    if config.enlarge:
      graphs = np.vstack([graphs, np.zeros([config.max_nodes-graphs.shape[0], graphs.shape[1]])])
      graphs = np.hstack([graphs, np.zeros([config.max_nodes, config.max_nodes-graphs.shape[1]])])
      if data["labels"] is not None:
        labels = np.array([l for l in data['labels'].values()])
        labels = np.vstack([labels, np.zeros([config.max_nodes-labels.shape[0], labels.shape[1]])])

    D_inverse, adj = normalization(graphs, False)

    X, initial_feature_channels = [], 0
    if data["labels"] is not None:
      X = sp.csr_matrix(labels)
      initial_feature_channels = config.vector_size
    D_inverse = to_tuple(D_inverse)
    adj = to_tuple(adj)
    X = to_tuple(X)
    dics[path] = (D_inverse, adj, Y[0], X, nodes_size_list, initial_feature_channels)
  
  with open(config.all_data_adj, "wb") as f_out:
      pickle.dump(dics, f_out)
  return dics

def contruct_all_graph(train_set,test_set,lbs_train, lbs_test, opcode_voc, features, contruct_rewrite = True):
  if os.path.exists('input/'+config.current_malware+'opcode/') :
    if contruct_rewrite is False:
       print('already constructed graphs')
       return
    else:
       print('remove repeat', 'input/'+config.current_malware+'opcode/')
       shutil.rmtree('input/'+config.current_malware+'opcode/')
  for _fn in train_set:
    G = contruct_graph(_fn, opcode_voc, features, 1)
    save_graph('input/'+config.current_malware+'opcode/'+'train/' +G['name']+'.json',G)
    #G = contruct_graph(_fn, opcode_voc, features, 0)
    #save_graph('input/'+config.current_malware+'bytes/'+'train/' +G['name']+'.json',G)
  for _fn in test_set:
    G = contruct_graph(_fn, opcode_voc, features, 1)
    save_graph('input/'+config.current_malware+'opcode/'+'test/' +G['name']+'.json',G)
    #G = contruct_graph(_fn, opcode_voc, features, 0)
    #save_graph('input/'+config.current_malware+'bytes/'+'test/' +G['name']+'.json',G)

def with_opcode(instructions, opcode_voc, features, labels, idx):
  if instructions == None:
    labels[idx] = features[opcode_voc[config.split_token]].tolist()#''.join([str(int(i)) for i in features[opcode_voc[config.split_token]]])
    #find_inverse_labels(inverse_labels,labels[idx],idx)
    return
  vectors = []
  for instruction in instructions.split('\n'):
    opcodes = instruction.split('\t')
    if (len(opcodes)==3):
      vectors.append(features[opcode_voc[Operand_normalization(opcodes[1],opcodes[2])]]) 
    else:
      print(instruction)
      vectors.append(features[opcode_voc[config.split_token]])
    #vectors.append(features[opcode_voc[config.split_token]])
  #labels[idx] = ''.join([str(int(i)) for i in np.sum(np.array(vectors), axis=0).tolist()])#''.join(vectors)
  #labels[idx] = sparse.csr_matrix(np.sum(np.array(vectors), axis=0))#''.join([str(int(i)) for i in np.sum(np.array(vectors), axis=0).tolist()])#''.join(vectors)
  labels[idx] = np.sum(np.array(vectors), axis=0).tolist()
  #find_inverse_labels(inverse_labels,labels[idx],idx)

def contruct_graph(fn, opcode_voc, features, feats):
  with open(fn, 'r') as jf:
    strs = json.load(jf)
    labels = {}
    block_id = {}
    for idx, block in enumerate(strs['blocks']):
      instructions = block[4]
      block_id[block[0]] = idx
      if feats == 1:
        with_opcode(instructions, opcode_voc, features, labels, idx)
      else:
        with_bytes(block, labels, idx)
    edges = []
    graph = np.zeros([len(labels), len(labels)], dtype=np.uint8)
    for edge in strs['edges']:
      edges.append([block_id[edge[0]],block_id[edge[1]]])
      graph[block_id[edge[0]]][block_id[edge[1]]] = 1
    if (len(edges)==0 or len(labels)==0):
      print('wrong file', fn)
      return
    return {'graph':graph.tolist(), 'nodes_size':len(labels),'labels':labels, 'edges': edges, 'target': return_label(fn), 'name':fn.split('/')[-1]}#,'inverse_labels':inverse_labels}

def Operand_normalization(op2, op3):
  return op2#, op3
  
def save_graph(fn, G):
  if os.path.exists(fn) :
    print('repeat', fn)
  os.makedirs(os.path.dirname(fn), exist_ok=True)
  with open(fn, 'w') as outfile:
      json.dump(G , outfile)
      #pickle.dump(G, outfile)

'''
def with_bytes(block, labels, idx):
  if block[3] != None:
    labels[idx]=list(map(float, list(str(bin(int(str(block[3]),16))[2:].zfill(config.vector_size_bytes)))))
  else:
    labels[idx]=list(map(float, list('0'*config.vector_size_bytes)))

def find_inverse_labels(inverse_labels, key, node):
  if key not in inverse_labels:
    inverse_labels[key] = [node]
  else:
    inverse_labels[key].append(node)
'''

# step 4
def normalization(adj, symmetric = False):
    adj = np.array(adj)
    adj = adj + np.eye(adj.shape[0])
    #adj = normalize_adj(adj, symmetric)
    D_inverse = np.linalg.inv(np.diag(np.sum(adj, axis=1)))
    '''
    if config.detector_type == "gcn":
      adj = normalize_adj(adj, symmetric)
    elif config.detector_type == "gnn":
      D_inverse = np.linalg.inv(np.diag(np.sum(adj, axis=1)))
    '''
    return sp.csr_matrix(D_inverse), sp.csr_matrix(adj)

def normalize_adj(adj, symmetric=False):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d)
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj)
    return a_norm

def create_input_data(path, dicts, symmetric=False):
    data = json.load(open(path))
    graphs, nodes_size_list, labels = data['graph'], data['nodes_size'], [data['target']]
    Y = [labels]

    D_inverse, adj = normalization(graphs, symmetric)
    X, initial_feature_channels = [], 0
    if data["labels"] is not None:
      X=sp.csr_matrix(label)
      initial_feature_channels = config.vector_size
    dicts[path] = (D_inverse, adj, Y[0], X, nodes_size_list, initial_feature_channels)
    return D_inverse, adj, Y[0], X, nodes_size_list, initial_feature_channels


def read_all_data_proc(rewrite, path=False):
  if rewrite == False:
    print(path)
    if path == False:
      with open(config.all_data_adj, 'rb') as f:
        return pickle.load(f)
    else:
      with open(path, 'rb') as f:
        return pickle.load(f)
  ending = '*.json'
  train_graph_paths = glob.glob(config.train_graph_folder+ending)
  test_graph_paths = glob.glob(config.test_graph_folder+ending)
  os.makedirs(os.path.dirname(config.all_data_adj), exist_ok=True)
  dics = {}
  #files = queue.Queue()
  for idx, path in enumerate(train_graph_paths+test_graph_paths):
    create_input_data(path, dics)
  with open(config.all_data_adj, "wb") as f_out:
      pickle.dump(dics, f_out)
  return dics
'''
    files.put(path)
  threads = []
  threads_num = files.qsize() if files.qsize() < MAX else MAX
  for j in range(threads_num):
    threads.append(MyThread(files, dics))
  for x in threads:
    x.start()  
  for t in threads:
    t.join()

class MyThread(threading.Thread):
  def __init__(self, files, dics):
    self.files = files
    self.dics = dics
    threading.Thread.__init__(self)

  def run(self):
    while True:
      if self.files.qsize() > 0:
        path = self.files.get()
        print('thread %s is running...' % threading.current_thread().name, self.files.qsize())
        prcs = multiprocessing.Process(target=create_input_data, args=(path, self.dics))
        prcs.start()
        prcs.join()
      else:
        break
'''
### Others
def list_to_vec(opcodes, features, opcode_voc):
  vectors = []
  for opcode in opcodes:
    opcode = opcode.lower()
    if opcode not in opcode_voc:
      print(opcode)
      return None
    vectors.append(features[opcode_voc[opcode]])
  return np.sum(np.array(vectors), axis=0)


'''
def construct_all_graph_new(data=['data/features/benign/', 'data/features/malware/']):
  files = []
  global all_files
  all_files = []
  scan_file(files, data)
  config.opcode_voc = 'all_graph_with_normal'
  os.makedirs(os.path.dirname(config.opcode_voc), exist_ok=True)
  files, opcode_voc = load_opcode_voc(True,files)
  idx = sorted(opcode_voc.values())
  eye = np.eye(max(idx) + 1)
  features = eye[idx]
  print(features.shape)
  print('Loaded...')

def read_all_data(rewrite):
  if rewrite == False and os.path.exists(config.all_data_adj):
    print('reloading...')
    with open(config.all_data_adj, 'rb') as f:
      return pickle.load(f)
  ending = '*.json'
  train_graph_paths = glob.glob(config.train_graph_folder+ending)
  test_graph_paths = glob.glob(config.test_graph_folder+ending)
  os.makedirs(os.path.dirname(config.all_data_adj), exist_ok=True)
  dics = {}
  for idx, path in enumerate(train_graph_paths+test_graph_paths):
    if idx % 20 == 0:
      print(idx, path)
    dics[path] = create_input_data(path)
  with open(config.all_data_adj, "wb") as f_out:
      pickle.dump(dics, f_out)
  return dics

def read_all_data_misclassify(graph_paths, detector, types, rewrite):
    os.makedirs(os.path.dirname(config.misclassify_data_adj+config.detection_model), exist_ok=True)
    if rewrite == False and os.path.exists(config.misclassify_data_adj+config.detection_model):
      print('reloading misclassify...')
      with open(config.misclassify_data_adj+config.detection_model, 'rb') as f:
        return pickle.load(f)
    dics = read_all_data(graph_paths, rewrite=rewrite)
    res = {}
    for path in graph_paths:
        D_inverse, A_tilde, Y, X, nodes_size_list, _ = dics[path]
        if (Y[0] == config.mal_label):
            _,_,pred = detection(types, detector, D_inverse, A_tilde, X, nodes_size_list, False)
            #print(Y[0],pred)
            if pred == config.mal_label:
              res[path] = dics[path]

    with open(config.misclassify_data_adj+config.detection_model, "wb") as f_out:
        pickle.dump(res, f_out)
    return res

'''

if __name__ == "__main__":
  #enlarge()
  train_set, val_set, test_set, lbs_train, lbs_val, lbs_test = split_data(rewrite=False)
  retrieve_all_graph('malware/final-3-all-btbv/opcode/gnn/all_data_adj_feature_matrix',train_set, val_set, test_set, lbs_train, lbs_val, lbs_test, False)

  #opcode_voc, features = extract_voc(train_set, val_set, test_set, lbs_train, lbs_val, lbs_test, False)
  #new_contruct_all_graph(train_set, val_set, test_set, lbs_train, lbs_val, lbs_test,  opcode_voc, features, False)
  #contruct_all_graph(train_set,test_set,lbs_train, lbs_test, opcode_voc, features, False)
  #read_all_data_proc(rewrite=False)
  

  
