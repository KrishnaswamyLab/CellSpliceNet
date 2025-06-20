import pandas as pd
from scipy.stats import skew, kurtosis
import argparse
import os
from args import parse_args  
   
def load_magic_file(args): 
    df_magic = pd.read_csv(
        args.pp01_impute_magic_outpath+'241119b_scCounts_neuronsBulk_magic_filtered.csv', 
        index_col=0
    )
    return df_magic

def neuron_set_dfs_extractor(df_magic):

    neuron_type_list = [header.split('_')[0] for header in df_magic.columns] 
    neuron_type_set = set(neuron_type_list)   

    # loop through each unique neuron
    neuron_set_dfs = {}
    for neuron_i in neuron_type_set:
        # filter columns that include the current neuron
        filtered_neuron_i = [col for col in df_magic.columns if col.startswith(neuron_i)]  
        # create a new DataFrame with the filtered neurons
        neuron_set_dfs[neuron_i] = df_magic[filtered_neuron_i]
 
    return neuron_set_dfs

def apply_neuron_moments(neuron_set_dfs):
    neuron_set_moments_dfs = {}  
    for neuron_i, neuron_i_df in neuron_set_dfs.items():  
        mean = neuron_i_df.mean(axis=1)
        variance = neuron_i_df.var(axis=1)
        skewness = neuron_i_df.apply(skew, axis=1)
        kurt = neuron_i_df.apply(kurtosis, axis=1) 
        # Create a new DataFrame to store the moments
        neuron_i_moments_df = pd.DataFrame({
            'mean': mean,
            'variance': variance,
            'skewness': skewness,
            'kurtosis': kurt
        })
        
        # Store the moments DataFrame in the dictionary
        neuron_set_moments_dfs[neuron_i] = neuron_i_moments_df
 
    return neuron_set_moments_dfs

def per_neuron_moment_save(args, neuron_set_moments_dfs):
    os.makedirs(args.pp02_compute_moments_outpath, exist_ok=True) 
    for key, moments_df in neuron_set_moments_dfs.items():
        filename = f"{key}_moments.csv"
        filepath = os.path.join(args.pp02_compute_moments_outpath, filename)
        moments_df.to_csv(filepath, index=True)

if __name__ == "__main__":   
    args = parse_args() 

    print('i) loading the output of the magic file') 
    df_magic = load_magic_file(args)  

    print('ii) extracting the neuron set-based df_magic')
    neuron_set_dfs = neuron_set_dfs_extractor(df_magic) 

    print('iii) applying moments on each neurons') 
    neuron_set_moments_dfs = apply_neuron_moments(neuron_set_dfs)

    print('iv) saving neuron-based moments')
    per_neuron_moment_save(args, neuron_set_moments_dfs)

 
    



































# import pandas as pd
# from scipy.stats import skew, kurtosis
# import argparse
# import os

 
# def parse_args():
#     parser = argparse.ArgumentParser(description="Compute statistical moments for expression data.") 

#     parser.add_argument('--input_file', default='../dataset/241119b_scCounts_neuronsBulk_magic_filtered.csv', type=str, help='Path to the input CSV file')
#     parser.add_argument('--output_dir', default='../dataset/pp02_moments/', type=str, help='Path to the output directory')
#     return parser.parse_args()

# args = parse_args()  

# df_magic = pd.read_csv(args.input_file, index_col=0)
# df_magic

# # Split each element in col_headers on '_' and take the first part
# split_headers = [header.split('_')[0] for header in df_magic.columns]

# # Find unique elements
# unique_headers = set(split_headers)

# # Print the unique elements
# print(unique_headers)
# print(f"Number of unique elements: {len(unique_headers)}")

# # Create a dictionary to store DataFrames for each unique header
# header_dfs = {}

# # Loop through each unique header
# for header in unique_headers:
#     # Filter columns that include the current header
#     filtered_cols = [col for col in df_magic.columns if col.startswith(header)]
    
#     # Create a new DataFrame with the filtered columns
#     header_dfs[header] = df_magic[filtered_cols]

# # Print the keys of the dictionary to verify
# print(header_dfs.keys())


# # Create a dictionary to store statistical moments for each DataFrame
# moments_dfs = {}

# # Loop through each key-value pair in header_dfs
# for header, df in header_dfs.items():
#     # Calculate mean, variance, skewness, and kurtosis for each row
#     mean = df.mean(axis=1)
#     variance = df.var(axis=1)
#     skewness = df.apply(skew, axis=1)
#     kurt = df.apply(kurtosis, axis=1)
    
#     # Create a new DataFrame to store the moments
#     moments_df = pd.DataFrame({
#         'mean': mean,
#         'variance': variance,
#         'skewness': skewness,
#         'kurtosis': kurt
#     })
    
#     # Store the moments DataFrame in the dictionary
#     moments_dfs[header] = moments_df

# # Print the keys of the dictionary to verify
# print(moments_dfs.keys())

# # Create an output directory
# output_dir = args.output_dir
# os.makedirs(output_dir, exist_ok=True)

# # Loop through each key-value pair in moments_dfs and save to CSV
# for key, moments_df in moments_dfs.items():
#     filename = f"{key}_moments.csv"
#     filepath = os.path.join(output_dir, filename)
#     moments_df.to_csv(filepath, index=True)