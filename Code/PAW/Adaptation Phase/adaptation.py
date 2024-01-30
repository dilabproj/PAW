import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss

import wandb
from utils import *
from sklearn.metrics import  cohen_kappa_score
from collections import Counter

fixed_random_seed(args.seed)
torch.set_num_threads(8)

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    if not args.not_use_wandb:
        wandb.log({"lr_running":param_group['lr']})
    return optimizer


def train_target(args):
    if not args.not_use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.project_name,
            name=args.name,
            settings=wandb.Settings(code_dir="."),
            # track hyperparameters and run metadata
            config={
            "subject": args.target_subject,
            "class div":args.gent,
            "class div for each source":args.gent_source,
            'gent_par':args.gent_par,
            'gent_source_par':args.gent_source_par,
            'confident ratio':args.confident_ratio
            }
        )
        config=wandb.config

    dset_loaders = get_partial_subjects_dataloaders()
    
    ##########################
    #### Model Initialize ####
    ##########################    
    netF_list=[]
    netC_list=[]
    print("Evaluation subjects ",args.evaluation_subjects)
    for i in args.evaluation_subjects:
        
        if i != args.target_subject:
            save_model_path = f"../../../source_pretrained_weight/training_phase/dd/{args.base_model}/{args.dataset}/{args.base_model}_s{i}.pt"
            model = get_model().to(args.device)
        
            model. load_state_dict(torch.load(save_model_path, map_location=args.device))
        
            if args.base_model=='eegnet':
                netF = model.feature
                netC = model.class_classifier
            elif args.base_model=='eegtcnet':
                netF = torch.nn.Sequential(model.EEGNet, model.TCN_block)
                netC = model.dense
        
            netF_list.append(netF)
            netC_list.append(netC) 
    
    netQ = network.DRL(args)
    
    #########################
    ####### Optimizer #######
    #########################
    param_group = []
    for i in range(args.source_num):
        netF_list[i].eval()
        for k, v in netF_list[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]

        netC_list[i].eval()
        for k, v in netC_list[i].named_parameters():
            v.requires_grad = False
        
    netQ.eval()
    for k, v in netQ.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]    

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    ######################
    ### Start Training ###
    ######################
    
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    for epoch in range(args.max_epoch):
        
        for i, (inputs_test, _, tar_idx) in enumerate(dset_loaders["target"]):

            if iter_num % interval_iter == 0 and args.cls_par > 0:
                
                for i in range(args.source_num):
                    netF_list[i].eval()
                netQ.eval()

                memory_label,  _, _, sample_weight = obtain_pseudo_label(dset_loaders['pl'], netF_list, netC_list, netQ, args)
                memory_label = torch.from_numpy(memory_label).to(args.device)

                for i in range(args.source_num):
                    netF_list[i].train()
                netQ.train()


            inputs_test = inputs_test.to(args.device).float()
            source_repre = torch.eye(args.source_num).to(args.device)

            iter_num += 1
            if not args.not_use_lrsch:
                lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

            outputs_all = torch.zeros(args.source_num, inputs_test.shape[0], args.n_classes)
            outputs_all_w = torch.zeros(inputs_test.shape[0], args.n_classes)
            init_ent = torch.zeros(1, args.source_num)

            for i in range(args.source_num):
                features_test = netF_list[i](inputs_test)
                outputs_test = netC_list[i](features_test) 
                softmax_prob = nn.Softmax(dim=1)(outputs_test)

                ent_loss = torch.mean(loss.Entropy(softmax_prob))
                init_ent[:, i] = ent_loss
                outputs_all[i] = outputs_test
            
            outputs_all = torch.transpose(outputs_all, 0, 1)
            
            source_repre_all=torch.zeros([outputs_all.shape[0], args.source_num, args.repre_num])
            weights_all=torch.zeros([outputs_all.shape[0],args.source_num])
            outputs_all_softmax = nn.Softmax(dim=2)(outputs_all)
            for idx,sample in enumerate(outputs_all_softmax):
                sample = sample.cpu().detach().numpy()
                source_repre = get_source_repre(sample, args).to(args.device)
                source_repre_all[idx] = source_repre
            weights_all=netQ(source_repre_all)
            
            for i in range(inputs_test.shape[0]):
                outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])
   
            weights_all = torch.transpose(weights_all, 0, 1)
            outputs_all = torch.transpose(outputs_all, 0, 1)
            
            pred = memory_label[tar_idx]
            batch_weight = sample_weight[tar_idx]
            
            ########################
            ### Loss Computation ###
            ########################
             
            if args.cls_par > 0:
                pseudo_loss = nn.CrossEntropyLoss(reduction='none')(outputs_all_w, pred.cpu().long())
                pseudo_loss = pseudo_loss * batch_weight
                pseudo_loss = torch.mean(pseudo_loss)
                classifier_loss = args.cls_par * pseudo_loss
            else:
                classifier_loss = torch.tensor(0.0)
            
            gent=0
            gent_source=0
            if args.ent:
                softmax_out = nn.Softmax(dim=1)(outputs_all_w)
                entropy_loss = torch.mean(loss.Entropy(softmax_out))
                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                    gent = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))*args.gent_par
                    entropy_loss -= gent
                    
                if args.gent_source:
                    source_softmax=nn.Softmax(dim=-1)(outputs_all)
                    source_softmax=source_softmax.mean(dim=1)
                    gent_source = torch.mean(-source_softmax * torch.log(source_softmax + 1e-5))*args.gent_source_par
                    entropy_loss -= gent_source
                    
                im_loss = entropy_loss * args.ent_par
                classifier_loss += im_loss

            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()
        
            if not args.not_use_wandb:
                wandb.log({"Loss": classifier_loss.data.cpu().numpy(),
                        "Pseudo label loss":pseudo_loss.data.cpu().numpy(),
                        "im loss":im_loss.data.cpu().numpy(),
                        "gent loss":gent.data.cpu().numpy(),
                        "gent soure loss":gent_source.data.cpu().numpy()})
            
            if iter_num % interval_iter == 0 or iter_num == max_iter:
                for i in range(args.source_num):
                    netF_list[i].eval()
                netQ.eval()
                acc, _, kappa = cal_acc_multi(dset_loaders['test'], netF_list, _, netC_list, netQ, args)
                if not args.not_use_wandb:
                    wandb.log({"Acc": acc,
                            "Kappa": kappa})   
                log_str = 'Iter:{}/{}; Accuracy = {:.2f}%'.format(iter_num, max_iter, acc)
                print(log_str + '\n')

    i=0
    for s in args.evaluation_subjects:
        if s!= args.target_subject:
            torch.save(netF_list[i].state_dict(),
                    osp.join(args.output_dir, f"s{s}_target_F.pt"))
            torch.save(netC_list[i].state_dict(),
                    osp.join(args.output_dir, f"s{s}_target_C.pt"))
            i+=1
    torch.save(netQ.state_dict(),
                osp.join(args.output_dir, f"s{s}_target_sw.pt"))              
            
    if not args.not_use_wandb:
        wandb.finish()

#########################
##### Pseudo label ######
#########################

def obtain_pseudo_label(loader, netF_list, netC_list, netQ, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader) 
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(args.device).float()
            source_repre = torch.eye(args.source_num).to(args.device) 

            outputs_all = torch.zeros(args.source_num, inputs.shape[0], args.n_classes)
            outputs_all_w = torch.zeros(inputs.shape[0], args.n_classes)

            for i in range(args.source_num):
                features = netF_list[i](inputs)
                outputs = netC_list[i](features)
                outputs_all[i] = outputs
            
            outputs_all = torch.transpose(outputs_all, 0, 1)
            
            source_repre_all=torch.zeros([outputs_all.shape[0], args.source_num, args.repre_num])
            weights_all=torch.zeros([outputs_all.shape[0],args.source_num])
            outputs_all_softmax = nn.Softmax(dim=2)(outputs_all)
            for idx,sample in enumerate(outputs_all_softmax):
                sample = sample.cpu().numpy()
                source_repre = get_source_repre(sample, args).to(args.device)
                source_repre_all[idx] = source_repre 
            weights_all =  netQ(source_repre_all)

            for i in range(inputs.shape[0]):
                outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])

            if start_test:
                all_output = outputs_all_w.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs_all_w.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    print("first sample weight:", np.array(weights_all[0]))
    print("last sample weight:", np.array(weights_all[-1]))
    all_output = nn.Softmax(dim=1)(all_output)
    _, pred_label = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(pred_label).float() == all_label).item() / float(all_label.size()[0])
    
    # Probability-based confidence anchor group
    all_prob = all_output.float().cpu().numpy()
    prob_max_id = all_prob.argsort(axis=1)[:, -1]
    prob_max2_id = all_prob.argsort(axis=1)[:, -2]

    prob_max = np.zeros(all_prob.shape[0])
    prob_max2 = np.zeros(all_prob.shape[0])
    for i in range(all_prob.shape[0]):
        prob_max[i] = all_prob[i, prob_max_id[i]]
        prob_max2[i] = all_prob[i, prob_max2_id[i]]
    prob_diff = prob_max - prob_max2

    prob_diff_tsr = torch.from_numpy(prob_diff).detach()
    
    weights=torch.ones(len(prob_diff_tsr))*args.low_confi_weight
    counter=Counter(pred_label.tolist())
    class_confi_num = int(all_prob.shape[0] * args.confident_ratio/len(counter))
    idx_prob_order = prob_diff_tsr.topk(int((all_prob.shape[0])), largest=True)[-1].cpu().numpy().tolist()
    
    # balance confi
    idx_confi_prob = []
    class_idx_dict={}
    confi_threshold=np.percentile(prob_diff, 25) #prob_diff.mean()
    for element, count in counter.items():
        class_idx=np.where(np.array(pred_label.tolist())==element)[0]
        class_idx_confi_ordr=[x for x in idx_prob_order if ((x in class_idx) and (prob_diff[x]>confi_threshold))]
        class_idx_confi = class_idx_confi_ordr[:class_confi_num]
        class_idx_dict[element]=class_idx_confi
        weights[class_idx_dict[element]] = 1
        print(f"class {element} get {len(class_idx_confi)} confident samples")
        if not args.not_use_wandb:
            wandb.log({f"class {element} num":len(class_idx_confi)})
        idx_confi_prob.extend(class_idx_confi)
    
    different_high_confi = set(idx_confi_prob).symmetric_difference(set(args.previous_confi_idx))
    args.confi_idx_all = list(set(args.confi_idx_all)|set(idx_confi_prob))
    # print(f"len confi idx all {len(args.confi_idx_all)}")
    # print(f"{len(different_high_confi)} different idx: {different_high_confi}")
    if not args.not_use_wandb:
        wandb.log({"num_different_high_confi":len(different_high_confi), "num confi idx all":len(args.confi_idx_all)})

    args.previous_confi_idx=idx_confi_prob
    
    acc_high_confi = torch.sum(torch.squeeze(pred_label[idx_confi_prob]).float() == all_label[idx_confi_prob]).item() / float(all_label[idx_confi_prob].size()[0])
    high_confi_confi = np.sum(prob_diff[idx_confi_prob])/len(idx_confi_prob)
    # print(f"acc_high_confi {acc_high_confi} ave high_confi_confi {high_confi_confi}")
    
    if not args.not_use_wandb:
        wandb.log({"high confi acc":acc_high_confi,"ave high confi confi":high_confi_confi})
    
    label_confi = None
    pred_label = pred_label.numpy()
    return pred_label.astype('int'), label_confi, all_label, weights

def cal_acc_multi(loader, netF_list, netB_list, netC_list, netQ, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(args.device).float()
            source_repre = torch.eye(args.source_num).to(args.device)

            outputs_all = torch.zeros(args.source_num, inputs.shape[0], args.n_classes)
            outputs_all_w = torch.zeros(inputs.shape[0], args.n_classes)

            for i in range(args.source_num):
                features = netF_list[i](inputs)
                outputs = netC_list[i](features)
                outputs_all[i] = outputs
            
            outputs_all = torch.transpose(outputs_all, 0, 1)
            
            source_repre_all=torch.zeros([outputs_all.shape[0], args.source_num, args.repre_num])
            weights_all=torch.zeros([outputs_all.shape[0],args.source_num])
            outputs_all_softmax = nn.Softmax(dim=2)(outputs_all)
            for idx,sample in enumerate(outputs_all_softmax):
                sample = sample.cpu().numpy()
                source_repre = get_source_repre(sample, args).to(args.device)
                source_repre_all[idx] = source_repre
            weights_all =netQ(source_repre_all)
            for i in range(inputs.shape[0]):
                outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])

            if start_test:
                all_outputs_all=outputs_all.float().cpu()
                all_output = outputs_all_w.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_outputs_all= torch.cat((all_outputs_all, outputs_all.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs_all_w.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    
    _, predict = torch.max(all_output, 1)
    accuracy_all = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    kappa = cohen_kappa_score(all_label, np.array(torch.squeeze(predict).float()))

    i=0
    for s in args.evaluation_subjects:
        if s!= args.target_subject:
            _, predict = torch.max(all_outputs_all[:,i,:], 1)
            accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
            print(f"s{s} acc {round(accuracy*100,2)}")
            if not args.not_use_wandb:
                wandb.log({f"s{s} acc":accuracy*100})
            i+=1    
        
    return accuracy_all * 100, mean_ent, kappa

if __name__ == "__main__":
    
    args.project_name = f"adaptation"
    args.source_num=len(args.evaluation_subjects)-1
    args.repre_num = args.n_classes
    
    for target in args.evaluation_subjects:
        args.output_dir = f"Result/{args.project_name}/{args.base_model}/{args.dataset}/{args.name}/t{target}"
                
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)                
        args.target_subject=target
        args.previous_confi_idx=[]
        args.confi_idx_all=[]
        print(f"args.gent {args.gent}")
        train_target(args)
