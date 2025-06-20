import argparse
import os

def mkdir_fun(path):
    if not os.path.exists(path): 
        os.mkdir(path)

def make_directroy(args):   
    mkdir_fun(args.pp01_impute_magic_outpath)       
    mkdir_fun(args.pp02_compute_moments_outpath) 
    mkdir_fun(args.pp03_build_graph_celltype) 
    mkdir_fun(args.pp04_get_scattering_coeffs)  
 
def parse_args():
    EXPRESSION_DATA_ROOT = '/gpfs/radev/home/aa2793/project/cellnn/dataset/Alternative-Splicing/expression_dataset/'
    
    parser = argparse.ArgumentParser(description="preprocessing of the expression data.") 
    parser.add_argument('--expression_data_root', default=EXPRESSION_DATA_ROOT, type=str, help='Path to the input CSV file')
    parser.add_argument('--scCounts_neuronsBulk_path', default=EXPRESSION_DATA_ROOT+'241119b_scCounts_neuronsBulk.mtx', type=str, help='Path to the input CSV file')
    parser.add_argument('--row_headers_path', default=EXPRESSION_DATA_ROOT+'241119b_scCounts_neuronsBulk.rownames', type=str, help='Path to the input CSV file')
    parser.add_argument('--col_headers_path', default=EXPRESSION_DATA_ROOT+'241119b_scCounts_neuronsBulk.colnames', type=str, help='Path to the input CSV file')
    parser.add_argument('--list_sf', default=EXPRESSION_DATA_ROOT+'list_sf.tsv', type=str, help='Path to the input CSV file')
    
    parser.add_argument('--pp01_impute_magic_outpath', default=EXPRESSION_DATA_ROOT+'pp01_magic/', type=str, help='Path to the input CSV file')
    parser.add_argument('--pp02_compute_moments_outpath', default=EXPRESSION_DATA_ROOT+'pp02_moments/', type=str, help='Path to the output directory')
    parser.add_argument('--pp03_build_graph_celltype', default=EXPRESSION_DATA_ROOT+'pp03_graphs/', type=str, help='Path to the output directory')
    parser.add_argument('--pp04_get_scattering_coeffs', default=EXPRESSION_DATA_ROOT+'pp04_scatter/', type=str, help='Path to the output directory')
    
    make_directroy(parser.parse_args())
    return parser.parse_args()
 
 