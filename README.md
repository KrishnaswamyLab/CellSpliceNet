 
# Pytorch implementation of CellSpliceNet: Interpretable Multimodal Modeling of Alternative Splicing Across Neurons in *C. elegans*
We introduce CellSpliceNet, an interpretable transformer-based multimodal deep learning framework designed to predict splicing outcomes across the neurons of C. elegans. By integrating four complementary data modalities, namely long-range genomic sequence, local regions of interest (ROIs) in the RNA sequence, secondary structure, and gene expression, CellSpliceNet captures the complex interplay of factors that influence splicing decisions within the cellular context. CellSpliceNet employs modality-specific transformer embeddings, incorporating structural representations guided by mutual information and scattering graph embeddings. To this end, a novel and carefully designed multimodal multi-head attention mechanism preserves the integrity of each modality while facilitating selective cross-modal interactions, notably allowing gene expression data to inform sequence and structural predictions. Attention-based pooling within each modality highlights biologically critical elements, such as canonical intron–exon splice boundaries and accessible single-stranded RNA loop structures within the exon.  

**Authors:** Arman Afrasiyabi, Jake Kovalic, Chen Liu, Egbert Castro, Alexis Weinreb, Erdem Varol, David M. Miller III,  Marc Hammarlund,  Smita Krishnaswamy 

[**Preprint (bioRxiv)**](https://www.biorxiv.org/content/10.1101/2025.06.22.660966v1) // 
[**Repository**](https://github.com/KrishnaswamyLab/CellSpliceNet)  // 
[**Dataset**](https://github.com/KrishnaswamyLab/CellSpliceNet-dataset)

## Repository structure
```
CellSpliceNet/
  src/
    data/           # datasets + dataloaders
    models/         # model definitions (transformers, heads, etc.)
    nn/             # modules 
    train.py/       # train/eval loops 
    utils/          # logging, seeding, misc utils
    viz/            # visulaziation tools for ploting
  pp/               # preprocessing/postprocessing assets (running is optional as the preprocessing data have been offered)
  requirements.txt | environment.yml
  LICENSE 
  README.md
```

### Requirements
- OS: <Enterprise Linux 8.10>  
- Python: 3.9.18 >  
- CUDA: 11.3 (if using GPU)
- PyTorch: 1.10.2

###  Dependencies:
All dependencies are available in [environment.yml] (https://github.com/KrishnaswamyLab/CellSpliceNet/blob/main/requirements.txt)
 
---

 
### Data: Download & prepare
1. Download the dataset: [CellSpliceNet-dataset](https://github.com/KrishnaswamyLab/CellSpliceNet-dataset)
2. Open src/args.py and set the dataset root path, e.g.:
```bash
dataset_root = "/path/to/your/dataset"
```

---

## Training and validation
```bash
python scr/train.py    
```
---
Our pretrained model is available at: [pretrained model](https://drive.google.com/drive/folders/1pVfKlGspW1sOB1W-rr9SQD_qas4u8uhy?usp=sharing)

## Citing
If you use this code or models, please cite:

```bibtex
@article{Afrasiyabi2025CellSpliceNet,
  title   = {CellSpliceNet: Interpretable Multimodal Modeling of Alternative Splicing Across Neurons in C. elegans},
  author  = {Arman Afrasiyabi, Jake Kovalic, Chen Liu, Egbert Castro, Alexis Weinreb, Erdem Varol, David M. Miller III,  Marc Hammarlund,  Smita Krishnaswamy },
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.06.22.660966},
  url     = {https://www.biorxiv.org/content/10.1101/2025.06.22.660966v1}
}
```
