import matplotlib.pyplot as plt
import torch
import pandas as pd
    
def gene_along_neuron_plots(metadata_dictionary, file_name = 'a.png', dpi=100):    
  
    gene_id_list = metadata_dictionary['gene_id_list']
    gene_name_list = metadata_dictionary['gene_name_list']
    neuron_name_list = metadata_dictionary['neuron_name_list'] 
    gene_id_set = metadata_dictionary['gene_id_set']
    
    targets = metadata_dictionary['targets']  
    preds = metadata_dictionary['preds']  
#     len_gene_id = len(gene_id_set)
      
    fig, axs = plt.subplots(5, 5, figsize=(20, 20))  # Create a 10x10 grid of subplots
    cmap = plt.get_cmap('tab20b')  

    # Generate 100 colors
    colors = [cmap(i) for i in range(25)]

    # Flatten the axes array for easy indexing
    axs = axs.flatten()
    counter = 0
    # Iterate over each gene identifier, ensure i < 100 to match the axes array
    for i, gene_i in enumerate(gene_id_set):
        if counter >= 25:  # Break the loop if there are more than 100 genes
            break
        
        ## mask the gene
        mask = torch.tensor([True if item == gene_i else False for item in gene_id_list])
        targets_i = targets[mask]
        preds_i = preds[mask]
        
        # mask the controls 
        indices = [idx for idx, m in enumerate(mask) if m.item() == True]
        gene_name_list_i = [gene_name_list[idx] for idx in indices]
 
        mask_se_i = torch.tensor([item.startswith('SE') for item in gene_name_list_i]) 

        if mask_se_i.shape[0]>2:
            targets_i = targets_i[mask_se_i]
            preds_i = preds_i[mask_se_i]

            if targets_i.shape[0]>2:    
                columns = ["GT_PSI", "PRED_PSI"]
                
                data_ut = [[i, j] for (i, j) in zip(targets_i, preds_i)]
                df_ut = pd.DataFrame(data_ut, columns=columns)
            
                # Plot using the specific axis in the grid
                axs[counter].scatter(df_ut["GT_PSI"], df_ut["PRED_PSI"], s=100, c='blue', alpha=0.4)

                # Optionally set labels for each subplot, or customize as needed
                axs[counter].set_xlabel("Truth")
                axs[counter].set_ylabel("Predicted") 
                axs[counter].set_xlim([-0.1, 1.1])
                axs[counter].set_ylim([-0.1, 1.1])
                axs[counter].set_title('Gene: '+gene_i[7:])  # Optionally add a title or customize each subplot
                
                # Remove top and right spines
                axs[counter].spines['top'].set_visible(False)
                axs[counter].spines['right'].set_visible(False)
                counter += 1 
  
    plt.tight_layout() 
    fig.savefig(file_name, dpi=dpi)
    plt.close(fig)
 
     
     
