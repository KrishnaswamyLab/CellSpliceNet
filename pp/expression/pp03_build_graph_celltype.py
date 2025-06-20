from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm
import pandas as pd
import sys
from multiprocessing import Pool, cpu_count
from args import parse_args  

def load_magic_file(args): 
    df_magic = pd.read_csv(
        args.pp01_impute_magic_outpath+'241119b_scCounts_neuronsBulk_magic_filtered.csv', 
        index_col=0
    )
    return df_magic

def build_mi_graph(neuron_i_df):
    neuron_i_df = neuron_i_df.T  
    # initialize an empty DataFrame to store mutual information values
    neuron_i_mi_matrix = pd.DataFrame(index=neuron_i_df.columns, columns=neuron_i_df.columns)  # [243 rows x 243 columns] 
    # calculate mutual information for each pair of columns
    for col1 in neuron_i_df.columns:
        for col2 in neuron_i_df.columns:
            if col1 != col2:  
                mi = mutual_info_regression(neuron_i_df[[col1]], neuron_i_df[col2]) 
                neuron_i_mi_matrix.loc[col1, col2] = mi[0] 
            else:
                neuron_i_mi_matrix.loc[col1, col2] = 0  # Mutual information with itself is 0 

    import pdb;pdb.set_trace()
    return neuron_i_mi_matrix

if __name__ == "__main__": 

    args = parse_args()  
    df_magic = load_magic_file(args) 
    neuron_sets = df_magic.columns.str.split('_').str[0].unique()

    def process_neuron(neuron_i):
        
        neuron_i_df = df_magic.filter(like=f"{neuron_i}_")
        print(f"Processing group: {neuron_i}")
         
        mi_matrix = build_mi_graph(neuron_i_df)
        
        # Save the mutual information matrix to a CSV file
        neuron_i_save_path = f"{args.pp03_build_graph_celltype}mi_matrix_{neuron_i}.csv"
        mi_matrix.to_csv(neuron_i_save_path)
        print(f"MI matrix for neuron '{neuron_i}' saved to {neuron_i_save_path}")

    with Pool(processes=cpu_count()) as pool: 
        for _ in tqdm(pool.imap_unordered(process_neuron, neuron_sets), total=len(neuron_sets)):
            pass

    # for neuron_i in neuron_sets:
    #     process_neuron(neuron_i)