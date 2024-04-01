import os
import io
import sys
from utils import *
import angr
import networkx as nx
from gnn_detector_april import GNN_Detector
from shutil import copyfile

import logging
logging.getLogger('angr').setLevel('CRITICAL')
with open(config.opcode_voc, 'r') as jf:
    opcode_voc = json.load(jf)
idx = sorted(opcode_voc.values())
eye = np.eye(max(idx) + 1)
features = eye[idx]

model_name='models/gnn-mc/'+config.current_malware
model = GNN_Detector(config.vector_size, config.top_k)
_, ckpt_manager = load_ckpt(model, model_name)#, 'models/gnn-mc/final-3-bbr/ckpt-21')
print(model_name) 

fi = random.randint(1,50000)
start = 700
save_log = io.open('result'+str(start)+'.log','a',encoding="utf-8")
dics = read_all_data_proc(rewrite=False)
names = []

for f in dics.keys():
    D_inverse, A_tilde, Y, X, nodes_size_list, _ = dics[f]
    names.append(f.split('/')[-1])
    #pos_score, logits, pre_y_value = detection('gnn', model, D_inverse, A_tilde, X, nodes_size_list, False, True)
    #save_log.write(f.split('/')[-1]+" "+str(nodes_size_list)+" "+str(pos_score.numpy())+"\n")
    #save_log.write(f+" "+str(nodes_size_list)+"\n")

def get_CFG(f):
    try:
        proj = angr.Project(f, load_options={'auto_load_libs':False})
    except:
        return None
    cfg = proj.analyses.CFGFast()
    block = proj.factory.block(proj.entry)
    blocks = {}
    G = nx.DiGraph()
    idx = 0
    labels = {}
    for n in cfg.graph.nodes():
        blocks[hex(n.addr)] = idx
        G.add_node(idx)
        idx += 1
        if n.block != None:
            block_instructions = n.block.capstone.__str__()
            with_opcode(block_instructions, opcode_voc, features, labels, idx) 
        else:
            with_opcode(None, opcode_voc, features, labels, idx)     
    nodes_size_list = len(labels)   
    for k, v in cfg.graph.edges():
        G.add_edge(blocks[hex(k.addr)], blocks[hex(v.addr)])

    graphs = np.zeros([config.max_nodes, config.max_nodes], dtype=np.uint8)
    for edge in G.edges:
        graphs[edge[0]][edge[1]] = 1
    labels = np.array([l for l in labels.values()])
    labels = np.vstack([labels, np.zeros([config.max_nodes-labels.shape[0], labels.shape[1]])])
    D_inverse, adj = normalization(graphs, False)
    X = sp.csr_matrix(labels)
    initial_feature_channels = config.vector_size
    D_inverse = to_tuple(D_inverse)
    D_inverse=(D_inverse[0].astype(np.float32),D_inverse[1],D_inverse[2])
    adj = to_tuple(adj)
    adj=(adj[0].astype(np.float32),adj[1],adj[2])
    X = to_tuple(X)
    X=(X[0].astype(np.float32),X[1],X[2])
    return D_inverse, adj, 1, X, nodes_size_list, initial_feature_channels

def get_rewards(f, name = 'gnn'):
    try:
        D_inverse, A_tilde, Y, X, nodes_size_list, _ = get_CFG(f)
    except:
        return None
    save_log.write(f+" new node: "+str(nodes_size_list)+'\n')
    pos_score, logits, pre_y_value = detection(name, model, D_inverse, A_tilde, X, nodes_size_list, False, True)
    return pos_score

import subprocess
def call_ida(input_file):
  script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orp-0.3/inp_ida.py")
  if not os.path.exists(script):
    print("error: could not find inp_ida.py (%s)" % script)
    sys.exit(1)
  command = 'idaq.exe -A -S"\\"' + script + '\\"" ' + input_file
  print("executing:", command)
  exit_code = subprocess.call(command)
  print("exit code:", exit_code)

def call_rand(input_file, seed, iters, _m):
  if seed == 5:
      script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ropf/orp.py")
      if not os.path.exists(script):
        print("error: could not find orp.py (%s)" % script)
        sys.exit(1)
      command = 'C:\Python27\python.exe ' + script + ' -m ' + input_file+' -s '+str(_m)+' -i ' + str(iters)
      print("executing:", command)
      exit_code = subprocess.call(command)
      print("exit code:", exit_code)
  else:
      script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orp-0.3/orp.py")
      if not os.path.exists(script):
        print("error: could not find orp.py (%s)" % script)
        sys.exit(1)
      command = 'C:\Python27\python.exe ' + script + ' -r ' + input_file+' -s '+str(seed)+' -n ' + str(iters)
      print("executing:", command)
      exit_code = subprocess.call(command)
      print("exit code:", exit_code)
  
def sai():
    ori = glob.glob('data/malware2/*')
    evade = 0
    total = 0
    diff = 0
    
    name = 'gnn'
    enum = 0
    checked = []
    with io.open('checked'+str(start)+'.log','r',encoding="utf-8") as f:
        for l in f.readlines():
            checked.append(l.strip('\n'))
    check_log = io.open('checked'+str(start)+'.log','a',encoding="utf-8")
    for i in range(start,start+100):
        f = ori[i]
        fn = f.split('\\')[-1]
        feat_loc = 'data/tmp3/'+fn
        os.makedirs('data/tmp3/', exist_ok=True)
        if fn in checked or os.path.exists(feat_loc):
            continue
        print('total '+str(total)+' enum '+str(enum)+" " + f+'\n')
        save_log.write('total '+str(total)+' enum '+str(enum)+" " + f+'\n')
        ori_logits = get_rewards(f, name)
        if ori_logits == None:
            continue
        save_log.write(f+" "+str(ori_logits.numpy())+'\n')
        enum += 1
        if np.argmax(ori_logits,-1)[0] == config.labels['ben']:
            save_log.write('error.....'+ f+'\n')
            print(f)
            continue
        ori_size = os.stat(f).st_size
        total += 1
        iterations = 0
        save_log.write('fname: '+f+' ori_logits: '+str(ori_logits.numpy())+'\n')
        check = 0
        copyfile(f, feat_loc)
        f2 = feat_loc
        wrong = 0
        _m = 1
        while (iterations < 199): 
            iterations += 1
            if (iterations > 10 and len(f2.split('_')) < 2) or (len(f2.split('_')) > 1 and iterations-int(f2.split('_')[-1])>25)  or (wrong>10):
                break
            #TODO IPR+Disp-5
            save_log.write('modifying.....'+ f2+'\n')
            for sample in glob.glob("data/tmp3/*.diff"):
                os.remove(sample)
            if os.path.exists(f2+".dmp.bz2"):
                os.remove(f2+".dmp.bz2")
            if os.path.exists("data/tmp3/"+'.'.join(fn.split('.')[0:-1])+".idb"):
                os.remove("data/tmp3/"+'.'.join(fn.split('.')[0:-1])+".idb")
            call_ida(f2)
            rest = random.randint(1,5)
            call_rand(f2, rest, iterations, _m*10)
            
            #TODO new address
            feat_loc = f2+"_"+str(iterations)
            try:
                logits = get_rewards(feat_loc, name)
            except:
                continue
            if logits == None:
                wrong += 1
                save_log.write('wrong!! '+feat_loc+" "+ f2+" "+f2+'\n')
                continue
            wrong = 0
            print(logits)
            save_log.write('fname: '+feat_loc+' new_logits: '+str(logits.numpy())+'\n')
            new_size = os.stat(feat_loc).st_size*1.
            
            diff += (new_size-ori_size)/ori_size
            save_log.write("orisize: "+str(ori_size)+" newsize:"+  str(new_size)+" diff:"+ str((new_size-ori_size)/ori_size)+" new logits:"+str(logits.numpy())+'\n')
            if diff > 0.05:
                print(diff, new_size, ori_size)
                break
            if np.argmax(logits,-1)[0] == config.labels['ben']:
                evade += 1
                save_log.write("evade!evade: "+str(evade)+" total: "+ str(total)+ " acc:"+ str(evade/total)+" size: "+ str(diff/total)+'\n')
                check = 1
                copyfile(feat_loc, 'data/succ2/'+feat_loc.split('/')[-1])
                break
            
            import pdb
            pdb.set_trace()
            if logits[0][1] >= ori_logits[0][1]:
                continue
            print('keep rest:', rest,'new loc', feat_loc)            
            save_log.write("keep rest:"+ str(rest)+'\n')
            if rest != 5:
                ori_logits = logits
                f2 = feat_loc
            else:
                _m += 1
        save_log.write("current: "+str(evade)+" total: "+ str(total)+ " acc:"+ str(evade/total)+" size: "+ str(diff/total)+'\n')
                
        checked.append(fn)
        check_log.write(fn+'\n')
    print(total, evade/total)

if __name__ == '__main__':
    sai()

