import sys
import pandas as pd
import numpy as np
import os
import csv
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.append(".")
from scatter import Scatter, scatter_moments
from args import parse_args  


def build_graph_continuous(adj_mat, moments_data):
    """
    Build a graph from the mutual information matrix.
    
    Parameters:
        adj_mat (pd.DataFrame): The mutual information matrix.
    
    Returns:
        torch_geometric.data.Data: The graph.
    """

    # adj_mat to torch tensor
    adj_mat = torch.tensor(adj_mat.values, dtype=torch.float32)

    # Extract edges and edge weights
    edge_index = np.vstack(np.nonzero(np.triu(adj_mat, k=1)))  # Get edges
    edge_weight = adj_mat[edge_index[0], edge_index[1]]        # Get weights

    # Convert to PyTorch tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long)    # Shape: [2, num_edges] 
    edge_weight = edge_weight.clone().detach().to(torch.float32)

    # node_features = torch.tensor(moments_data.values, dtype=torch.float32)
    node_features = torch.diag(torch.tensor(moments_data['mean'].values, dtype=torch.float32)) 
    graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)

    return graph


def scatter_graph(graph):
    """
    Computes the scattering coefficients for a given graph.

    Parameters:
    graph (torch_geometric.data.Data): The input graph data object containing node features `x`.

    Returns:
    torch.Tensor: The scattering coefficients of the input graph.
    """
    in_channels = graph.x.shape[-1]
    max_graph_size = graph.x.size(0)
    
    scattering = Scatter(in_channels, max_graph_size)
    scatter_coeffs = scattering(graph)
    
    return scatter_coeffs


def main():

    args = parse_args() 
    mi_graph_dir = args.pp03_build_graph_celltype
    moments_dir = args.pp02_compute_moments_outpath

    mi_neuron_names_list = os.listdir(mi_graph_dir)

    for mi_neuron_i in tqdm(mi_neuron_names_list, desc="Processing mi_neuron_names_list"):
        file_path = os.path.join(mi_graph_dir, mi_neuron_i)
        if os.path.isfile(file_path):
            
            mi_graph_data = pd.read_csv(file_path, index_col=0)                     # [243, 243]
            neuron_i = mi_neuron_i.split('_')[-1].split('.')[0]                     # 'OLL'
            moments_file_path = os.path.join(moments_dir, f"{neuron_i}_moments.csv")
             
            exact_order_of_sp = mi_graph_data.index.tolist()
            with open(args.expression_data_root+'exact_order_of_sp.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(exact_order_of_sp)
            
            # import pdb;pdb.set_trace()
            # exact_order_of_sp = pd.DataFrame(exact_order_of_sp, index=index) 
            # exact_order_of_sp.index.to_series().to_csv('exact_order_of_sp.csv', index=False, header=False)
 
            if os.path.isfile(moments_file_path):
                moments_data = pd.read_csv(moments_file_path, index_col=0)            # 243, 4
            else:
                print(f"Moments mi_neuron_i not found for {mi_neuron_i}")

            graph = build_graph_continuous(
                adj_mat=mi_graph_data, 
                moments_data=moments_data
            )     # graph: Data(x=[243, 243], edge_index=[2, 24542], edge_attr=[24542])
            scatter_coeffs = scatter_graph(graph)         # scatter_coeffs.shape = [243, 11, 243]
            print(scatter_coeffs.shape)
  
            output_dir = args.pp04_get_scattering_coeffs
            os.makedirs(output_dir, exist_ok=True)
            base_filename = mi_neuron_i.split('.')[0].split('_')[-1]
            output_file = os.path.join(output_dir, f"scatter_coeffs_{base_filename}.pt")
            torch.save(scatter_coeffs, output_file)

            # Process the data as needed
            print(f"Processed {mi_neuron_i}")

if __name__ == "__main__":
    main()




# def main():

#     args = parse_args() 
#     graph_dir = args.pp03_build_graph_celltype
#     moments_dir = args.pp02_compute_moments_outpath

#     files = os.listdir(graph_dir)

#     for file in tqdm(files, desc="Processing files"):
#         file_path = os.path.join(graph_dir, file)
#         if os.path.isfile(file_path):
#             data = pd.read_csv(file_path, index_col=0)
#             label = file.split('_')[-1].split('.')[0]
#             moments_file_path = os.path.join(moments_dir, f"{label}_moments.csv")
#             if os.path.isfile(moments_file_path):
#                 moments_data = pd.read_csv(moments_file_path, index_col=0)
#             else:
#                 print(f"Moments file not found for {file}")
#             # graph = build_graph_kernel(data, moments_data)
#             graph = build_graph_continuous(data, moments_data)
#             scatter_coeffs = scatter_graph(graph)
  
#             output_dir = args.pp04_get_scattering_coeffs
#             os.makedirs(output_dir, exist_ok=True)
#             base_filename = file.split('.')[0].split('_')[-1]
#             output_file = os.path.join(output_dir, f"scatter_coeffs_{base_filename}.pt")
#             torch.save(scatter_coeffs, output_file)

#             # Process the data as needed
#             print(f"Processed {file}")

# TODO: Maybe needed for non-dirac node features
# data_path = "../data/expression/counts/240919_scCounts_neuronsBulk_magic_filtered.csv"
# data = pd.read_csv(data_path)

# def build_graph_kernel(adj_mat, moments_data, sigma=1.0):
#     """
#     Build a graph from the mutual information matrix.
    
#     Parameters:
#         adj_mat (pd.DataFrame): The mutual information matrix.
    
#     Returns:
#         torch_geometric.data.Data: The graph.
#     """
#     # Use Gaussian kernel to compute edge weights
#     adj_mat = torch.exp(- torch.tensor(adj_mat.values) ** 2 / (2 * sigma ** 2))
#     # Extract edges and edge weights
#     edge_index = np.vstack(np.nonzero(np.triu(adj_mat, k=1)))  # Get edges
#     edge_weight = adj_mat[edge_index[0], edge_index[1]]  # Get weights

#     # Convert to PyTorch tensors
#     edge_index = torch.tensor(edge_index, dtype=torch.long)  # Shape: [2, num_edges]
#     edge_weight = torch.tensor(edge_weight, dtype=torch.float32)  # Shape: [num_edges]

#     # node_features = torch.tensor(moments_data.values, dtype=torch.float32)
#     node_features = torch.diag(torch.tensor(moments_data['mean'].values, dtype=torch.float32))
#     # import pdb; pdb.set_trace()
#     graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)

#     return graph

# def build_graph(adj_mat, moments_data, cutoff=0.35):
#     """
#     Build a graph from the mutual information matrix.
    
#     Parameters:
#         adj_mat (pd.DataFrame): The mutual information matrix.
    
#     Returns:
#         torch_geometric.data.Data: The graph.
#     """
#     adj_mat = adj_mat.applymap(lambda x: 1 if float(x) > cutoff else 0)
#     edge_index, _ = dense_to_sparse(torch.tensor(adj_mat.values, dtype=torch.float32))

#     node_features = torch.diag(torch.tensor(moments_data['mean'].values, dtype=torch.float32))
#     graph = Data(x=node_features, edge_index=edge_index)

#     return graph