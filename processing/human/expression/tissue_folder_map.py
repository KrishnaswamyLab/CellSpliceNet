"""
Map GTEx exported_counts folder prefixes (before '--GRCh38--gencode29')
to ASCOT / gtex_psi.csv.gz tissue column names (exact strings for CSV + .pt filenames).
"""
from __future__ import annotations


# Keys: directory basename prefix (first segment before "--")
# Values: exact column header in gtex_psi.csv.gz
FOLDER_TO_ASCOT: dict[str, str] = {
    "Adipose_-_Subcutaneous": "Subcutaneous - Adipose",
    "Adipose_-_Visceral_Omentum": "Visceral (Omentum) - Adipose",
    "Adrenal_Gland": "Adrenal Gland",
    "Artery_-_Aorta": "Aorta - Artery",
    "Artery_-_Coronary": "Coronary - Artery",
    "Artery_-_Tibial": "Tibial - Artery",
    "Brain_-_Amygdala": "Amygdala - Brain",
    "Brain_-_Anterior_cingulate_cortex_BA24": "Anterior cingulate - Brain",
    "Brain_-_Caudate_basal_ganglia": "Caudate nucleus - Brain",
    "Brain_-_Cerebellar_Hemisphere": "Cerebellar Hemisphere - Brain",
    "Brain_-_Cerebellum": "Cerebellum - Brain",
    "Brain_-_Cortex": "Cortex - Brain",
    "Brain_-_Frontal_Cortex_BA9": "Frontal Cortex - Brain",
    "Brain_-_Hippocampus": "Hippocampus - Brain",
    "Brain_-_Hypothalamus": "Hypothalamus - Brain",
    "Brain_-_Nucleus_accumbens_basal_ganglia": "Nucleus accumbens - Brain",
    "Brain_-_Putamen_basal_ganglia": "Putamen - Brain",
    "Brain_-_Spinal_cord_cervical_c-1": "Spinal cord (C1) - Brain",
    "Brain_-_Substantia_nigra": "Substantia nigra - Brain",
    "Breast_-_Mammary_Tissue": "Mammary Tissue - Breast",
    "Cells_-_Cultured_fibroblasts": "Xform. fibroblasts - Cells",
    "Cells_-_EBV-transformed_lymphocytes": "EBV-xform lymphocytes - Cells",
    "Colon_-_Sigmoid": "Sigmoid - Colon",
    "Colon_-_Transverse": "Transverse - Colon",
    "Esophagus_-_Gastroesophageal_Junction": "Gastroesoph. Junc. - Esophagus",
    "Esophagus_-_Mucosa": "Mucosa - Esophagus",
    "Esophagus_-_Muscularis": "Muscularis - Esophagus",
    "Heart_-_Atrial_Appendage": "Atrial Appendage - Heart",
    "Heart_-_Left_Ventricle": "Left Ventricle - Heart",
    "Kidney_-_Cortex": "Cortex - Kidney",
    "Liver": "Liver",
    "Lung": "Lung",
    "Minor_Salivary_Gland": "Minor Salivary Gland",
    "Muscle_-_Skeletal": "Skeletal - Muscle",
    "Nerve_-_Tibial": "Tibial - Nerve",
    "Ovary": "Ovary",
    "Pancreas": "Pancreas",
    "Pituitary": "Pituitary",
    "Prostate": "Prostate",
    "Skin_-_Not_Sun_Exposed_Suprapubic": "Not Sun Exposed - Skin",
    "Skin_-_Sun_Exposed_Lower_leg": "Sun Exposed (Lower leg) - Skin",
    "Small_Intestine_-_Terminal_Ileum": "Ileum - Small Intestine",
    "Spleen": "Spleen",
    "Stomach": "Stomach",
    "Testis": "Testis",
    "Thyroid": "Thyroid",
    "Uterus": "Uterus",
    "Vagina": "Vagina",
    "Whole_Blood": "Whole Blood",
}


def ascot_tissue_list() -> list[str]:
    return sorted(set(FOLDER_TO_ASCOT.values()))
