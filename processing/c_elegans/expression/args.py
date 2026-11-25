import argparse
import os

def mkdir_fun(path):
    os.makedirs(path, exist_ok=True)

def make_directroy(args):   
    mkdir_fun(args.pp01_impute_magic_outpath)       
    mkdir_fun(args.pp02_compute_moments_outpath) 
    mkdir_fun(args.pp03_build_graph_celltype) 
    mkdir_fun(args.pp04_get_scattering_coeffs)  
 
def parse_args():
    """Defaults: inputs from <data_root>/c_elegans/raw, intermediates in
    <data_root>/c_elegans/processed/expression/work, output in .../expression/scatter_243."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from processing.common import default_data_root
    root = default_data_root() / 'c_elegans'
    RAW = str(root / 'raw') + '/'
    EXPR = str(root / 'raw' / 'expression') + '/'
    WORK = str(root / 'processed' / 'expression' / 'work') + '/'
    OUT = str(root / 'processed' / 'expression' / 'scatter_243') + '/'

    parser = argparse.ArgumentParser(description="preprocessing of the expression data.")
    parser.add_argument('--expression_data_root', default=WORK, type=str, help='Working directory (pp04 writes exact_order_of_sp.csv here)')
    parser.add_argument('--scCounts_neuronsBulk_path', default=EXPR+'241119b_scCounts_neuronsBulk.mtx', type=str, help='Single-cell counts (Matrix Market)')
    parser.add_argument('--row_headers_path', default=EXPR+'241119b_scCounts_neuronsBulk.rownames', type=str, help='Gene names (rows of the .mtx)')
    parser.add_argument('--col_headers_path', default=EXPR+'241119b_scCounts_neuronsBulk.colnames', type=str, help='Cell names (columns of the .mtx)')
    parser.add_argument('--list_sf', default=RAW+'list_sf.tsv', type=str, help='Splice-factor gene list')

    parser.add_argument('--pp01_impute_magic_outpath', default=WORK+'pp01_magic/', type=str, help='MAGIC-imputed counts')
    parser.add_argument('--pp02_compute_moments_outpath', default=WORK+'pp02_moments/', type=str, help='Per-neuron gene moments')
    parser.add_argument('--pp03_build_graph_celltype', default=WORK+'pp03_graphs/', type=str, help='Per-neuron mutual-information graphs')
    parser.add_argument('--pp04_get_scattering_coeffs', default=OUT, type=str, help='Scattering coefficients (model input)')

    args = parser.parse_args()
    make_directroy(args)
    return args
