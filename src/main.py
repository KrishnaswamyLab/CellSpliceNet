import torch
import os
import torch.nn as nn
import torch.optim as optim
from args import argparser_fn
import torch.optim.lr_scheduler as lr_scheduler
from data.splicedata_dataloader import splicedata_dataloader
from args import argparser_fn, print_gpu_info, make_directroy
from viz.training_scatter import scatter_plot
from utils.validation import validation_for_model, model_perd 
from model import model_registry
import argparse
import json
import numpy as np
import shutil, time
 

def main(replicate_status='replicate', batch_size=16, train_continue=False, model_key=''):
 
    args = argparser_fn(replicate_status, batch_size, server='misha')

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    data = splicedata_dataloader(args)
    data.setup()
    train_dataloader = data.train_dataloader()
    valid_dataloader = data.valid_dataloader()
    test_dataloader = data.test_dataloader()

    args = data.setup_hparams(args)

    # Initialize model
    ModelClass = model_registry.str2model(args.model) 
    model = ModelClass(args)

    # -------------------------
    # Wrap in DataParallel here
    # -------------------------
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs via nn.DataParallel.")
        model = nn.DataParallel(model)

    model.to(device)
    if train_continue: 
        model_key = args.model+'--'+model_key+'-'+replicate_status  
        if replicate_status == 'single':
            model_key = model_key[:-7]
        else: 
            model_key = model_key[:-10]

        model_key_model = '../../outputs/'+args.model+'/'+model_key+'-01Feb2025_'+replicate_status+'/'   # 
        model_label='best_validation_model' 
        model = torch.load(model_key_model+'model/'+model_label+'.pth', weights_only=False) 
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        # import pdb;pdb.set_trace() 
        with open(model_key_model+'model/loss_dict.json', 'r') as file: 
            loss_dict = json.load(file) 
        
        first_epoch_index = 90
        for _ in range(first_epoch_index):  
            scheduler.step()
        args.output_dir = '../../outputs/'+args.model+'/'+model_key+'-01Feb2025_'+replicate_status+'/' 
        args.output_key = model_key
    else:
        args.output_key = time.strftime(args.model + "--%Y%m%d-%H%M%S-") 
        # Prepare output directory
        args.output_dir = make_directroy(args)
        shutil.copytree('../../cellsplice.net/', args.output_dir + '/codes/', dirs_exist_ok=True)
        # Set up optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        first_epoch_index = 0 
        loss_dict = {'train_loss': [], 'valid_loss': [], 'epoch': None}
    print_gpu_info(args)

    best_valoss = np.inf
    prt_msg = 'epoch:%3d | train loss: %4.3f | valid loss: %4.3f | GPU: %.2f MB'

    # Training Loop
    for epoch in range(first_epoch_index, args.n_epochs):
        model.train()
        trlosseslist = []

        for iteration, (metadata, data_in, targets) in enumerate(train_dataloader):
            preds, loss_i, _ = model_perd(
                model, data_in, metadata, targets, device, embed_list_return=False
            ) 
            optimizer.zero_grad()
            gpu_usage_mb = torch.cuda.memory_allocated() / 1024**2  # memory in MB
            loss_i.backward()
            optimizer.step()
            trlosseslist.append(loss_i.item()) 

        if epoch%10==0:
            # Validation
            model.eval()
            valid_loss, preds_valid, targets_valid = validation_for_model(
                model,
                test_dataloader,
            )

            # Example scatter plot
            scatter_plot(
                gt_tensor=targets_valid,
                pred_tensor=preds_valid,
                path=args.output_dir + '/scatter_valid/psi/psi_at_ep_' + str(epoch),
                title='',
                label='',
                dpi=100,
                scatter_color='#0077b6',
                title_add=True
            )

            train_loss = np.mean(trlosseslist)
            print(prt_msg % (epoch + 1, train_loss, valid_loss, gpu_usage_mb))

            # Checkpoint best model
            if valid_loss < best_valoss:
                best_valoss = valid_loss
                # When using DataParallel, the actual model is in model.module
                if train_continue: 
                    torch.save(model.to('cpu'), args.output_dir + '/model/best_validation_model_train_continue.pth')
                else:
                    torch.save(model.to('cpu'), args.output_dir + '/model/best_validation_model.pth')

            # Save losses
            loss_dict['train_loss'].append(train_loss)
            loss_dict['valid_loss'].append(valid_loss) 
            loss_dict['epoch'] = epoch
            with open(args.output_dir+'/model/'+'loss_dict.json', "w") as file:
                json.dump(loss_dict, file) 

            # Optionally step the scheduler
            scheduler.step()  
            
            torch.save(model.to('cpu'), args.output_dir + '/model/epoch_'+str(epoch)+'_validation_model.pth')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', default=16, type=int, help=' ')
    parser.add_argument('--seed_value', default=42, type=int,   help='ff')
 
    parser.add_argument('--train_continue', default=True)  
    
    parser.add_argument('--replicate_status', default='replicate', type=str, help=' ') 
    # parser.add_argument('--model_key', default='20250404-195923')  
    parser.add_argument('--model_key', default='20250404-213118')  

    # parser.add_argument('--replicate_status', default='single', type=str, help=' ') 
    # parser.add_argument('--model_key', default='20250404-021039')  
 
    args = parser.parse_args() 
    torch.manual_seed(args.seed_value)
    np.random.seed(args.seed_value)

    main(
        replicate_status=args.replicate_status, 
        batch_size=args.batch_size,
        train_continue=args.train_continue,
        model_key=args.model_key,
    )

    



     
# if __name__ == "__main__":
#     batch_size = 16
#     # dataset_type = 'replicate'
#     # dataset_type = 'single' 
#     main(dataset_type='replicate', batch_size=16)
 
# import torch 
# import numpy as np 
# from viz.training_scatter import scatter_plot
# from model import model_registry
# from data.splicedata_dataloader import splicedata_dataloader
# from args import argparser_fn, print_gpu_info, make_directroy, extention
# from viz.training_gene_along_neuron_plots import gene_along_neuron_plots
# from utils.validation import validation_for_model, model_perd
# from torch.optim import lr_scheduler 
# import json
# import shutil
# import time

# torch.manual_seed(1337)
# np.random.seed(42)

# if __name__ == "__main__":  
#     batch_size = 16           
#     dataset_type = '01Feb2025_neuron_replicate'  
 
#     args = argparser_fn(dataset_type, batch_size, server='misha')
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
#     data = splicedata_dataloader(args)
#     data.setup()
#     train_dataloder = data.train_dataloader() 
#     valid_dataloder = data.valid_dataloader() 
#     test_dataloder = data.test_dataloader()
    
#     args = data.setup_hparams(args) 
#     model = model_registry.str2model(args.model) 
#     args.output_key = time.strftime(args.model+"--%Y%m%d-%H%M%S") 
    
#     args.output_dir = make_directroy(args)  
#     shutil.copytree('../../cellsplice.net/', args.output_dir+'/codes/', dirs_exist_ok=True)   
    
#     model = model(args)   
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
#     loss_dict = {
#         'train_loss': [],
#         'valid_loss': [],
#     }
    
#     print_gpu_info(args.output_key)
#     best_valoss = np.inf  
#     prt_meg = 'epoch:%3d  | train loss: %4.3f  | valid loss: %4.3f  | GPU: %.2f MB'
#     model.to(device) 

#     for epoch in range(args.n_epochs): 
#         trlosseslist = []
#         model.train()    
#         for iteration, (metadata, data, targets) in enumerate(train_dataloder):     
#             preds, loss_i, _ = model_perd(model, data, metadata, targets, device, embed_list_return=False)  
#             optimizer.zero_grad()  
#             gpu_usage_mb = torch.cuda.memory_allocated() / 1024**2  # memory in MB 
#             loss_i.backward()   
#             optimizer.step()    
#             trlosseslist.append(loss_i.item())       
        
#         model.eval()   
#         valid_loss, preds_valid, targets_valid = validation_for_model(
#             model, 
#             test_dataloder,  
#         )  

#         scatter_plot(
#             gt_tensor=targets_valid, pred_tensor=preds_valid, 
#             path=args.output_dir+'/scatter_valid/psi/psi_at_ep_'+str(epoch), 
#             title='', label='', dpi=100, scatter_color='#0077b6', 
#             title_add=True
#         )
        
#         train_loss = sum(np.array(trlosseslist))/len(trlosseslist)  
#         print(prt_meg % (epoch + 1, train_loss, valid_loss, gpu_usage_mb))  

#         if valid_loss < best_valoss:  
#             best_valoss = valid_loss   
#             torch.save(model.to('cpu'), args.output_dir+'/model/best_validation_model.pth')  

#         # save loss loss dictionary 
#         loss_dict['train_loss'].append(train_loss)
#         loss_dict['valid_loss'].append(valid_loss)  
        
#         # save current model  
#         torch.save(model.to('cpu'), args.output_dir+'/model/current_model.pth')
#         if epoch % 5 ==0:
#             torch.save(
#                 model.to('cpu'), 
#                 args.output_dir+'/model/current_model_epoch('+str(epoch)+').pth'
#             )   
#             with open(args.output_dir+'/model/'+'loss_dict.json', "w") as file:
#                 json.dump(loss_dict, file) 
        
#         scheduler.step()  
#         model.to(device)  