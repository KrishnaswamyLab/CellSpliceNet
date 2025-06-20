import pandas as pd
from scipy.io import mmread
from scipy.io import mmwrite
import magic
from args import parse_args
import os

def load_count_matrix(args):
    mtx_data = mmread(args.scCounts_neuronsBulk_path)
    row_headers = pd.read_csv(args.row_headers_path, header=None).squeeze().tolist()
    col_headers = pd.read_csv(args.col_headers_path, header=None).squeeze().tolist()
    return mtx_data, row_headers, col_headers

def apply_magic(mtx_data, row_headers, col_headers):  
    save_file_name = '241119b_scCounts_neuronsBulk_magic.csv'
    magic_data_path = args.pp01_impute_magic_outpath + save_file_name
     
    if os.path.exists(magic_data_path):  
        df_magic = pd.read_csv(magic_data_path) 
    else:
        magic_operator = magic.MAGIC()
        df = pd.DataFrame(mtx_data.toarray(), index=row_headers, columns=col_headers) 
        df_magic = magic_operator.fit_transform(df)
        output_csv_file = data_root+'241119b_scCounts_neuronsBulk_magic.csv'
        df_magic.to_csv(output_csv_file)
    return df_magic
 
def filter_splicefactors(args, df_magic): 
    save_file_name = '241119b_scCounts_neuronsBulk_magic_filtered.csv'
    filtered_magic_path = args.pp01_impute_magic_outpath + save_file_name
    if os.path.exists(filtered_magic_path): 
        df_magic_filtered = pd.read_csv(filtered_magic_path) 
    else: 
        sf_df = pd.read_csv(args.list_sf, sep='\t')  
        filtered_genes = sf_df['gene_id']
        df_magic_filtered = df_magic.loc[df_magic.index.intersection(filtered_genes)] 
        df_magic_filtered.to_csv(filtered_magic_path)

    return df_magic_filtered


if __name__ == "__main__":  
    
    args = parse_args() 

    print('   i) loading mtx file and rows and col names file') 
    mtx_data, row_headers, col_headers = load_count_matrix(args)

    print('   ii) applying magic on the data')
    df_magic = apply_magic(mtx_data, row_headers, col_headers)  
 
    print('   iii) filtering the sf') 
    df_magic_filtered = filter_splicefactors(args, df_magic)         # [243 rows x 33692 columns]
 

 


# df_magic_filtered
#          Unnamed: 0     AVH_1     RIA_2     AVH_3      I5_4  ...  CEP_33687  CEP_33688  CEP_33689  CEP_33690  AVH_33691
# 0    WBGene00017929  0.036728  0.023382  0.014723  0.002978  ...   0.005836   0.027679   0.013368   0.082302   0.101484
# 1    WBGene00021938  0.035094  0.028125  0.047275  0.015681  ...   0.013838   0.006659   0.003420   0.041492   0.030084
# 2    WBGene00021921  0.152046  0.175386  0.122996  0.027293  ...   0.017909   0.048894   0.052853   0.241129   0.115632
# 3    WBGene00018776  0.048399  0.032613  0.029901  0.002164  ...   0.018631   0.047213   0.020906   0.198147   0.133752
# 4    WBGene00022301  0.003845  0.003146  0.004566  0.000314  ...   0.001896   0.003976   0.001482   0.025395   0.015204
# ..              ...       ...       ...       ...       ...  ...        ...        ...        ...        ...        ...
# 238  WBGene00006964  0.026157  0.011866  0.019315  0.006140  ...   0.008280   0.001643   0.001256   0.074291   0.024622
# 239  WBGene00011051  0.025827  0.030051  0.012618  0.007094  ...   0.000796   0.003513   0.002172   0.066844   0.034810
# 240  WBGene00003392  0.007378  0.011781  0.006427  0.000249  ...   0.001069   0.001712   0.003265   0.036638   0.015230
# 241  WBGene00011043  0.002442  0.000872  0.005915  0.000040  ...   0.000660   0.000276   0.000990   0.006053   0.008213
# 242  WBGene00013160  0.002090  0.001420  0.006029  0.000052  ...   0.003202   0.001621   0.004039   0.015219   0.011761

# [243 rows x 33692 columns]

 