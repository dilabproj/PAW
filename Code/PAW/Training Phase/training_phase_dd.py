import numpy as np
#ignore warning 
from warnings import simplefilter
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from utils import *
from domain_discriminator import Discriminator

fixed_random_seed(0)
torch.set_num_threads(4)

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=DeprecationWarning)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="EEGNet independent model save")
    
    # parameter about wandb
    # parser.add_argument('--project', type=str, default='Training_phase_ablation')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--not_use_wandb',action='store_true')# if set, it will be true. or false

    # parameter about training
    parser.add_argument('--base_model',type=str, default='eegnet',choices=['eegnet','eegtcnet'])
    parser.add_argument('--dataset', type=str, default='2a',choices=['2a','2b'])
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu_id', type=str, default='0')    
    parser.add_argument('--use_normalization',action='store_true')
    parser.add_argument('--use_bandpass',action='store_true')
    parser.add_argument('--no_gpu', action='store_true')
        
    # parameter about dd
    parser.add_argument('--use_domain_label',type=bool, default=True)
    parser.add_argument('--weight_dd',type=float, default=0.05)
    
    # parameter about learning rate scheduler
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--sch_factor', type=float, default=0.9)
    parser.add_argument('--sch_patience', type=int, default=10)
    parser.add_argument('--sch_cooldown', type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.use_gpu = torch.cuda.is_available()
    if args.no_gpu:
        args.device = torch.device("cpu")
        
    save_dir=f"model_weight/dd/{args.base_model}/{args.dataset}/{args.name}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    
    get_dataset_setting(args)

    for target in args.evaluation_subjects:
        
        fixed_random_seed(0)
        args.target_subject = target
        
        train_loader = get_train_loader(args)
        x_test_list, y_test_list = get_subjects_data_list(args)
        
        model = get_model(args)
        dd = Discriminator(input_size=args.feat_dim, num_classes=len(args.train_sessions)).to(args.device)
          
        if not args.not_use_wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project=f"Training_phase_{args.base_model}_{args.dataset}",
                name=args.name,
                settings=wandb.Settings(code_dir="."),
                # track hyperparameters and run metadata
                config={
                "target subject": target,
                "init_lr": args.init_lr,
                "max epochs":args.epochs,
                "sch_factor":args.sch_factor,
                "sch_patience":args.sch_patience,
                "sch_cooldown":args.sch_cooldown,
                "weight_dd":args.weight_dd
                }
            )
            config=wandb.config
            wandb.watch(model)

        # Init loss
        criterion = nn.CrossEntropyLoss().to(args.device)
        domain_criterion = nn.NLLLoss().to(args.device)
        
        # Init optimizer
        optimizer_model = optim.Adam(model.parameters(), lr=args.init_lr)
        D_opt = torch.optim.Adam(dd.parameters())
        
        # Init scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_model, mode='min', factor=args.sch_factor, 
                                                   patience=args.sch_patience, cooldown=args.sch_cooldown,verbose=True)

        # train model
        for epoch in range(args.epochs):
            
            model.train()
            dd.train()
            running_loss = []
            running_domain_loss = []
            for i, (data, labels, domain_label) in enumerate(train_loader):
                
                data=data.to(args.device,dtype=torch.float)
                labels=labels.to(args.device,dtype=torch.long)
                domain_label = domain_label.to(args.device,dtype=torch.long)
                
                ########################################
                ### Training of domain descriminator ###
                ########################################
                model.eval()
                dd.train()
                
                outputs, feature = model(data)
                output_domain=dd(feature)
                
                Ld = domain_criterion(output_domain, domain_label)
                optimizer_model.zero_grad()
                D_opt.zero_grad()
                Ld.backward()
                D_opt.step()                
                
                ################################################
                # Training of feature extractor and classifier #
                ################################################
                model.train()
                dd.eval()
                
                outputs, feature = model(data)
                output_domain=dd(feature)
                
                Ld = domain_criterion(output_domain, domain_label)
                Lc = criterion(outputs, labels)
                lamda = args.weight_dd * get_lambda(epoch, args.epochs)
                Ltot = Lc -lamda*Ld
                
                # backpropagation
                optimizer_model.zero_grad()
                D_opt.zero_grad()
                
                Ltot.backward()
                
                optimizer_model.step()

                running_loss.append(Ltot.data.cpu().numpy())
                running_domain_loss.append(Ld.data.cpu().numpy())
            
            model.eval() 
            
            mean_loss=float(sum(running_loss)/len(running_loss))
            mean_domain_loss=float(sum(running_domain_loss)/len(running_domain_loss))
             
            scheduler.step(mean_loss)

            
            if epoch %5 ==0:

                for i, subject in enumerate(args.evaluation_subjects):
                    if subject == args.target_subject:
                        acc, kappa = metrics_computation(x_test_list[i], y_test_list[i], model, args)
                        
                if not args.not_use_wandb:
                    wandb.log({"train loss": mean_loss,"domain loss":mean_domain_loss, "lr": optimizer_model.state_dict()['param_groups'][0]['lr'],
                               "Target acc": acc})
                                        
                print('Epoch: ', epoch ,'train loss:', round(mean_loss,4), '  Train acc: ', round(acc*100,2),
                      'lr: ', optimizer_model.state_dict()['param_groups'][0]['lr'])
            
            if epoch==(args.epochs-1):
                
                for i, subject in enumerate(args.evaluation_subjects):
                    acc, kappa = metrics_computation(x_test_list[i], y_test_list[i], model,args)
                    if subject != args.target_subject:
                        print(f"Subject{subject} acc {round(acc*100,2)}")
                        if not args.not_use_wandb:
                            wandb.config.update({f"S{subject} acc": acc*100})
                
                torch.save(model.state_dict(),f"{save_dir}/{args.base_model}_s{target}.pt")
                if not args.not_use_wandb:
                    wandb.finish()
                break