import numpy as np
import torch
import random
from load_data import my_loader
from config import *
from model_EEGNet import EEGNet
from model_eegtcnet import EEGTCNet

def fixed_random_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    
def get_dataset_setting():
    if args.dataset == "2a":
        args.n_classes=4
        args.n_channels = 22
        args.input_window_samples = 1000
        if args.base_model=='eegnet':
            args.feat_dim = 16*int(args.input_window_samples/32)
        elif args.base_model=='eegtcnet':
            args.feat_dim=12
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
            
    elif args.dataset == "2b":
        args.n_classes=2
        args.n_channels = 3
        args.input_window_samples = 1000
        if args.base_model=='eegnet':
            args.feat_dim = 16*int(args.input_window_samples/32)
        elif args.base_model=='eegtcnet':
            args.feat_dim=12
            # exit()
        args.session_num = 5     
        args.total_sessions = 4
        args.n_subjects = 9   
        if args.base_model=='eegnet':
            args.evaluation_subjects=[3,4,7,8,9]
        elif args.base_model=='eegtcnet':
            args.evaluation_subjects=[2,3,4,7,8,9]        
    else:
        print('dataset unknown!')
        exit()

    return args

def get_partial_subjects_dataloaders():
    
    assert args.target_subject in args.evaluation_subjects, "Target subject not in evaluation subjects"
    
    if args.dataset=='2a':
        train_sessions=['T','E']
        test_sessions=['E']
        data_path='../../../../BNCI_2a/'
        
    elif args.dataset=='2b':
        train_sessions=[str(x) for x in range(1,6)]
        test_sessions=[str(x) for x in range(4,6)]
        data_path='../../../../BCICIV2b/'
    else:
        print('NOT Implement Yet!')
        exit()

    x_test=[]
    y_test=[]
    
    x_val=[]
    y_val=[]
    
    print("Test data:")
    for session in test_sessions:
        print(f"s{args.target_subject} {session}")
        x=np.load(data_path+'data/'+str(args.target_subject)+session+'.npy')
        y=np.load(data_path+'label/'+str(args.target_subject)+session+'.npy')
        
        x_test.append(x)
        y_test.append(y)
        
        if args.paradigm=='offline':
            x_val.append(x)
            y_val.append(y)
        elif args.paradigm=='online':
            x_val.append(x[:int(len(x)/2),:,:])
            y_val.append(y[:int(len(x)/2)])

    x_test=np.concatenate([sample for sample in x_test]).astype(np.float64)
    y_test=np.concatenate([sample for sample in y_test]).astype(np.float64)
    
    x_val=np.concatenate([sample for sample in x_val]).astype(np.float64)
    y_val=np.concatenate([sample for sample in y_val]).astype(np.float64)
    
    dataset_test = my_loader(x_test, y_test, if_test=True)
    test_loader= torch.utils.data.DataLoader(dataset=dataset_test,batch_size=args.batch_size,shuffle=False)

    dataset_val = my_loader(x_val, y_val)
    val_loader= torch.utils.data.DataLoader(dataset=dataset_val,batch_size=args.batch_size,shuffle=True)    
    
    dataset_pl = my_loader(x_val, y_val)
    pseudo_label_loader= torch.utils.data.DataLoader(dataset=dataset_pl,batch_size=args.batch_size,shuffle=False)    
    
    loaders={}
    loaders['target'] = val_loader
    loaders['test'] = test_loader
    loaders['pl'] = pseudo_label_loader
    
    return loaders

def get_model():
    
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

def get_source_repre(sample, args):
    source_repre_shape=[len(args.evaluation_subjects)-1,args.n_classes]
    source_repre = torch.zeros(source_repre_shape).to(args.device)
    for source_num in range(len(args.evaluation_subjects)-1):
        repre = torch.from_numpy(sample[source_num]).to(args.device)
        source_repre [source_num] = repre

    return source_repre


get_dataset_setting()


if __name__=="__main__":
    
    pass