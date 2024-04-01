#ben_label = 1
#mal_label = 0

# gat detector settings
hid_units = [3, 3] # numbers of hidden units per each attention head in each layer
n_heads = [3, 1, 1] # additional entry for the output layer
withmatmul = True
withattention = False#True

# gnn settings
GRAPH_CONV_LAYER_CHANNEL = 32
CONV1D_1_OUTPUT = 16
CONV1D_2_OUTPUT = 32
CONV1D_1_FILTER_WIDTH = GRAPH_CONV_LAYER_CHANNEL * 3 
CONV1D_2_FILTER_WIDTH = 5
DENSE_NODES = 128
LEARNING_RATE_DECAY = 0.99
top_k = 1000
max_gradient_norm = 1.0

# gcn settings
with_pooling = True
featureless = False
hidden1 = 128
hidden2 = 64
hidden3 = 32
hidden4 = 16

# generator setttings
l2_coef = 0.000001  # weight decay
penlty = 2
detection_model = 'gnn'
surrogate_type = 'gnn'
insert_nodes_num= 1
random_chose = 5
node_layers = 8
max_T = 30
top_k2 = 1250
reward = 2.0
#insert_code = ['xor','cmp','setne','mov','mov','push','call']
'''
insert_dead_opcode_seq = [
    ['nop'],['inc','push','dec','dec'],['mov'],['mov','add','mov'],
    ['mov','push','dec'],['mov','cmp','setg','movzx','mov','mov'],
    ['mov','add','imul','cmp','mov'],['XCHG','XCHG'],['mov','xor'],
    ['fcom'], ['fnclex'], ['fst'], ['fist'], ['fimul'], ['fidiv'], ['ficom'], ['fisubr'], ['fisub'], ['ficomp'], ['fidivr'], ['int'], ['fldcw'], ['fnstenv'], ['fldenv'], ['lock add'], ['fnstcw'], ['frstor'], ['fnsave'], ['rep stosb'], ['repe cmpsb'], ['fcmovnu'], ['fdivp'], ['bswap'], ['fucomi'], ['bt'], ['fxch'], ['fcmovu'], ['fcmovnb'], ['fmulp'], ['ffree'], ['faddp'], ['fucomip'], ['stosw'], ['fcmovb'], ['lock xor'], ['fcomi'], ['lock or'], ['lock dec'], ['fcmovne'], ['fucomp'], ['fcmovnbe'], ['fcmovbe'], ['fdivrp'], ['fsubrp'], ['lock xadd'], ['fcomip'], ['lock adc'], ['setg'], ['movups'], ['lock inc'], ['fucom'], ['lock and'], ['andps'], ['lock sbb'], ['fcmove'], ['cwd'], ['movsw'], ['paddb'], ['bts'], ['setl'], ['lock sub'], ['setb'], ['cmovo'], ['cmovns'], ['movhps'], ['cmovno'], ['setge'], ['lock xchg'], ['xadd'], ['repe cmpsd'], ['shld'], ['movaps'], ['seto'], ['cmovb'], ['fldz'], ['cvtdq2ps'], ['shrd'], ['movq'], ['sysenter'], ['cmovbe'], ['fchs'], ['cmovae'], ['movd'], ['paddd'], ['punpckhbw'], ['pcmpeqb'], ['syscall'], ['repe scasb'], ['repne scasd'], ['cmove'], ['pcmpgtb'], ['cmovl'], ['wbinvd'], ['psubb'], ['frndint'], ['iret'], ['cmpxchg'], ['unpcklps'], ['movlps'], ['xorps'], ['fninit'], ['femms'], ['subps'], ['fabs'], ['setle'], ['repne movsb'], ['sidt'], ['fcos'], ['pavgb'], ['orps'], ['pcmpeqw'], ['seta'], ['btr'], ['unpckhps'], ['por'], ['pmulhuw'], ['cmovne'], ['pshufw'], ['punpcklwd'], ['addps'], ['cvttps2pi'], ['paddsb'], ['rcpps'], ['cmovs'], ['cmovle'], ['punpcklbw'], ['maxps'], ['psubusb'], ['repne stosd'], ['sqrtps'], ['psubsb'], ['bnd jae'], ['pause'], ['fincstp'], ['rsqrtps'], ['cmovge'], ['cvtps2pi'], ['packssdw'], ['cvtps2pd'], ['psadbw'], ['repne cmpsb'], ['pcmpgtd'], ['comiss'], ['pandn'], ['punpckhdq'], ['cmovg'], ['fprem'], ['cmovnp'], ['cmova'], ['bnd jne'], ['pmuludq'], ['fcompp'], ['mulps'], ['aam'], ['rep lodsb'], ['andnps'], ['repne movsd'], ['fldpi'], ['bnd je'], ['setae'], ['btc'], ['pcmpgtw'], ['ffreep'], ['bnd jo'], ['psubq'], ['paddusb'], ['bsf'], ['bnd jg'], ['psubw'], ['punpckhwd'], ['cmovp'], ['repne stosb'], ['rdtsc'], ['bnd jge'], ['jcxz'], ['fld1'], ['packuswb'], ['setnp'], ['psubusw'], ['bnd jl'], ['emms'], ['pslld'], ['pinsrw'], ['shufps'], ['pxor'], ['repe scasd'], ['rep lodsd'], ['sets'], ['ucomiss'], ['pcmpeqd'], ['repne cmpsd'], ['paddusw'], ['pavgw'], ['movntps'], ['fldlg2'], ['psubsw'], ['cvtpi2ps'], ['packsswb'], ['bnd jle'], ['minps'], ['bnd jbe'], ['setbe'], ['punpckldq'], ['pand'], ['bnd ja'], ['setns'], ['scasw'], ['bnd jnp'], ['psubd'], ['paddq'], ['lodsw'], ['bnd jns'], ['maskmovq'], ['psllw'], ['bnd jb'], ['bnd jmp'], ['cmpps'], ['psllq'], ['f2xm1'], ['movnti'], ['setno'], ['psrad'], ['paddw'], ['divps'], ['pushf'], ['pmaxsw'], ['psraw'], ['pminub'], ['pmullw'], ['ftst'], ['bnd ret'], ['repne scasw'], ['cpuid'], ['psrld'], ['bsr'], ['paddsw'], ['fsincos'], ['bnd jno'], ['fldl2e'], ['aad'], ['cmpsw'], ['setp'], ['pmaxub'], ['bnd js'], ['cbw'], ['psrlw'], ['fscale'], ['popf'], ['fsin'], ['fyl2x'], ['psrlq'], ['pminsw'], ['fldln2'], ['fxam'], ['fldl2t'], ['fpatan'], ['fucompp'], ['pmulhw'], ['pmaddwd'], ['fptan'], ['fyl2xp1'], ['sgdt'], ['fsqrt'], ['movntq'], ['fprem1'], ['fxtract'], ['prefetchw'], ['bnd jp'], ['movlhps'], ['prefetcht0'], ['lock not'], ['movmskps'], ['prefetchnta'], ['lidt'], ['prefetcht1'], ['pextrw'], ['pmovmskb'], ['phaddsw'], ['lgdt'], ['clflush'], ['fxsave'], ['cmpxchg8b'], ['rep movsw'], ['stmxcsr'], ['ldmxcsr'], ['cvtpd2pi'], ['fxrstor'], ['int1'], ['into'], ['movdqa'], ['repe cmpsw'], ['addpd'], ['arpl'], ['phaddd'], ['outsb'], ['palignr'], ['insd'], ['enter'], ['psignb'], ['insb'], ['cvtsi2ss'], ['cmpunordps'], ['lock bts'], ['pmulhrsw'], ['outsd'], ['movntpd'], ['andnpd'], ['les'], ['retf'], ['movlpd'], ['movbe'], ['unpcklpd'], ['punpcklqdq'], ['pmaddubsw'], ['pshuflw'], ['cvttpd2pi'], ['lock cmpxchg'], ['bound'], ['sqrtsd'], ['pabsb'], ['lock btc'], ['cmpnleps'], ['cvtps2dq'], ['andpd'], ['cvtsi2sd'], ['lzcnt'], ['movapd'], ['unpckhpd'], ['cmpeqps'], ['fbld'], ['cmpnltps'], ['cvttpd2dq'], ['hsubps'], ['cmpordps'], ['pshufhw'], ['pshufd'], ['pshufb'], ['minpd'], ['psignd'], ['lds'], ['subsd'], ['rep lodsw'], ['mulsd'], ['movddup'], ['cmpss'], ['divpd'], ['cvtss2sd'], ['cvtss2si'], ['addsd']]
'''

insert_dead_opcode_seq = [ 
    ['inc','push','dec','dec'],['neg','neg'],['push','not','pop'],['add','sub'],['nop'],['not', 'not'],
    ['xor', 'xor', 'xor', 'xor', 'xor', 'xor'],['sub'], ['sub', 'add'],  ['add'], 
    ['XCHG', 'XCHG'],  ['CMOVNO'], ['CMOVNP'], ['CMOVA'], 
    ['mov','add','mov'],['mov','cmp','setg','movzx','mov','mov'],['PUSH', 'POP'], ['XCHG'], ['CMOVG'],['MOV'], 
    ['CMOVS'],  
    ['CMOVNS'], ['CMOVO'], ['CMOVL'], 
    ['BSWAP', 'BSWAP'], ['LEA'],  ['CMOVP'], ['PUSHF', 'POPF']
    #['CMOVNL'], ['byterev', 'byterev'], ['bitrev', 'bitrev'], ['CMOVNC'], ['CMOVNA'], ['CMOVNG'], ['CMOVC'], ['CMOVZ'], 
] 

'''
##gnn
insert_dead_opcode_seq = [ 
    ['inc','push','dec','dec'],['neg','neg'],['push','not','pop'],['nop'],['not', 'not'],
    ['xor', 'xor', 'xor', 'xor', 'xor', 'xor'],['add'], 
    ['CMOVNP'], ['CMOVA'], ['mov','cmp','setg','movzx','mov','mov'], ['XCHG'], ['CMOVG'],['MOV'], 
    ['CMOVS'],  ['CMOVNS'], ['CMOVO'], ['CMOVL'], 
    ['BSWAP', 'BSWAP'], ['LEA'],  ['CMOVP']
    #['CMOVNL'], ['byterev', 'byterev'], ['bitrev', 'bitrev'], ['CMOVNC'], ['CMOVNA'], ['CMOVNG'], ['CMOVC'], ['CMOVZ'], 
] 
'''

##gcn  
#
#       
opcode_substitution =[
    [['add'],['sub']],[['cmp'],['cmp','cmp','cmp']],
    [['lea'],['mov']],[['push'],['mov']],[['pop'],['mov']],
    [['sub'],['neg','add']]
]
insert_opaque_seq = [['mov','mov','imul','test','jns']]
cell_type = 'lstm'
is_birnn = False

# other parameters
batch_size =  100
max_nodes = 3300
learning_rate = 0.00300
epoch=10
split_token = 'None'
dropout = 0.5
max_manipulation = 0.05

enlarge = True
#current_malware, vector_size = 'final-3-bbr'+str(enlarge)+'/',548 #'final-4-bvrn/', 548#'final-5/', 469#'final-3/', 527#'final-2/',519#,#'all1/', 527#'all/', 514## 'all_no_normal/',514###'Trojan/',502#'Backdoor/', 507# 'Virus/', 410##
current_feats = 'opcode/'
vector_size_bytes = 3289
data_dir_benign = ["data/features/benign/"]

#current_malware, vector_size = 'final-3-all-btbvr/',595
#current_malware, vector_size = 'final-3-all-btvr/',595
#current_malware, vector_size = 'final-3-all-btr/',595
#current_malware, vector_size = 'final-3-all-btb/',595
#data_dir_malware = ["data/features/malware/"+'Trojan/',"data/features/malware/"+'Backdoor/', "data/features/malware/"+'Virus/', "data/features/malware/"+'Ransomware/']
#labels = {'ben':0, 'troj': 1, 'back': 2, 'virus': 3, 'rans':4}#}
current_malware, vector_size = 'final-3-all-btbv/',569
data_dir_malware = ["data/features/malware/"+'Trojan/',"data/features/malware/"+'Backdoor/', "data/features/malware/"+'Virus/']
labels = {'ben':0, 'troj': 1, 'back': 1, 'virus': 1}#}
#labels = {'ben':0, 'troj': 1, 'virus': 2, 'rans':3}
#current_malware, vector_size = 'final-3-all-btr/',595
#labels = {'ben':0, 'troj': 1, 'rans':2}#0.919#0.919#0.919
#labels = {'ben':0, 'troj': 1, 'back':2}
'''
current_malware, vector_size = 'final-3-bbr/',548
data_dir_malware = ["data/features/malware/"+'Backdoor/', "data/features/malware/"+'Ransomware/']#,"data/features/malware/"+'Virus/']#,"data/features/malware/"+'Worm-samples/']#]#]
labels = {'ben':0, 'back': 1, 'rans':2}#, 'virus': 3}#, 'rans': 4}#, 'worm': 2}#, {'ben':0,  'back': 2, 'worm': 3} }
'''

detector_type = 'gnn'
num_classes = len(set(labels.values()))
data_lens = "malware/"+current_malware+"data.lens.txt"
#all_data_adj = "malware/"+current_malware+current_feats+detector_type+"/all_data_adj_feature"
#all_data_adj_enlarge = "malware/"+current_malware+current_feats+detector_type+"/all_data_adj_feature_enlarge"
#misclassify_data_adj = "malware/"+current_malware+current_feats+detector_type+"/misclassify_data_adj_feature"
all_data_adj = "malware/"+current_malware+current_feats+"all_data_adj_feature"
all_data_adj_enlarge = "malware/"+current_malware+current_feats+"all_data_adj_feature_enlarge"
misclassify_data_adj = "malware/"+current_malware+current_feats+"misclassify_data_adj_feature"
opcode_voc = "malware/"+current_malware+"opcode_voc.json"
train_graph_folder = 'input/'+current_malware+current_feats+'train/'
test_graph_folder = 'input/'+current_malware+current_feats+'test/'



add_ri_gene = True

