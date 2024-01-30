import numpy as np
import torch
import random
from load_data import *
from model_EEGNet import EEGNet
from model_eegtcnet import EEGTCNet
import torch.nn.functional as F
from sklearn.metrics import  cohen_kappa_score, accuracy_score

###############################
###### Training Setting #######
###############################
def fixed_random_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

def get_dataset_setting(args):
    if args.dataset == "2a":
        args.n_classes=4
        args.n_channels = 22
        args.input_window_samples = 1000
        if args.base_model=='eegnet':
            args.feat_dim = 16*int(args.input_window_samples/32)
        elif args.base_model == 'eegtcnet':
            args.feat_dim = 12 # FT
        else:
            print('feat_dim unknown!')
            # exit()
        args.session_num = 2
        args.total_sessions = 2
        args.n_subjects = 9
        if args.base_model=='eegnet':
            args.evaluation_subjects=[1,2,3,4,5,6,9]
        elif args.base_model=='eegtcnet':
            args.evaluation_subjects=[3,4,5,6,7,9]
        
        args.train_sessions=['T','E']
        args.test_sessions=['E']
        args.data_path='../../../../BNCI_2a/'
        
    elif args.dataset == "2b":
        args.n_classes=2
        args.n_channels = 3
        args.input_window_samples = 1000
        if args.base_model=='eegnet':
            args.feat_dim = 16*int(args.input_window_samples/32)
        elif args.base_model == 'eegtcnet':
            args.feat_dim = 12 # FT
        else:
            print('feat_dim unknown!')
            # exit()
        args.session_num = 5     
        args.total_sessions = 4
        args.n_subjects = 9   
        if args.base_model=='eegnet':
            args.evaluation_subjects=[3,4,7,8,9]
        elif args.base_model=='eegtcnet':
            args.evaluation_subjects=[2,3,4,7,8,9]  
        
        args.train_sessions=[str(x) for x in range(1,6)]
        args.test_sessions=[str(x) for x in range(4,6)]
        args.data_path='../../../../BCICIV2b/'      
    else:
        print('dataset unknown!')
        exit()

    return args

def get_model(args):
    
    if args.base_model=='eegnet':
        init_model = EEGNet(
            args.n_channels,
            args.n_classes,
            input_window_samples=args.input_window_samples
        ).to(args.device)    

    elif args.base_model=='eegtcnet':
        
        F1 = 8
        KE = 32
        KT = 4
        L = 2
        FT = 12
        pe = 0.2
        pt = 0.3

        init_model = EEGTCNet(nb_classes = args.n_classes,Chans=args.n_channels, Samples=args.input_window_samples,
                       layers=L, kernel_s=KT,filt=FT, dropout=pt, activation='elu', 
                       F1=F1, D=2, kernLength=KE, dropout_eeg=pe).to(args.device)           

    return init_model

################################
####### Data Loaders ###########
################################

def get_train_loader(args):
    
    '''
    取得subject的dataloader，可設定是否使用session number
    '''
    get_dataset_setting(args)
    
    assert args.target_subject in args.evaluation_subjects, f"Target subject not in evaluation subjects: target {args.target_subject} evaluation subjects {args.evaluation_subjects}"
    
    x_train = []
    y_train = []
    s_train = []
    
    for i,session in enumerate(args.train_sessions):

        x=np.load(args.data_path+'data/'+str(args.target_subject)+session+'.npy')
        y=np.load(args.data_path+'label/'+str(args.target_subject)+session+'.npy')
        
        x_train.append(x)
        y_train.append(y)
        s_train.append(np.ones(len(x))*i) # session number

    x_train=np.concatenate([sample for sample in x_train])
    y_train=np.concatenate([sample for sample in y_train])    
    s_train=np.concatenate([sample for sample in s_train])    

    if args.use_domain_label:
        dataset=loader_with_domain_label(x_train, y_train, s_train)
    else:
        dataset=dependent_loader(x_train, y_train)
    train_loader= torch.utils.data.DataLoader(dataset=dataset,batch_size=args.batch_size,shuffle=True)
    
    return train_loader

# 把資料彙總已進行一次性的效能檢測
def get_subjects_data_list(args):
    get_dataset_setting(args)

    x_test_list=[]
    y_test_list=[]
    
    for subject in args.evaluation_subjects:
        
        x_test=[]
        y_test=[]        
        
        for session in args.test_sessions:
            x=np.load(args.data_path+'data/'+str(subject)+session+'.npy')
            y=np.load(args.data_path+'label/'+str(subject)+session+'.npy')
            
            x_test.append(x)
            y_test.append(y)

        x_test=np.concatenate([sample for sample in x_test])
        y_test=np.concatenate([sample for sample in y_test])    
        
        x_test_list.append(x_test)
        y_test_list.append(y_test)

    return x_test_list, y_test_list

def metrics_computation(x, y, model, args):
    
    pred = np.argmax (F.softmax(model(torch.from_numpy(x).float().to(args.device))[0],dim=-1).cpu().detach().numpy(), axis=-1)
    
    acc = accuracy_score(pred, y)
    kappa = cohen_kappa_score(pred, y)
    
    return acc, kappa

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.

fixed_random_seed(0)
if __name__=="__main__":
    
    pass