import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd

class dataloader(Dataset):
    def __init__(self, data_csv, sequence_mod_data, dataset_type): #, gene_embed_dict):#, auxiliary_data):

        self.dataset_type = dataset_type
        # data files
        self.data_csv = data_csv
        # sequence
        self.prime_seq_dict = sequence_mod_data["primary_sequence"]
        self.sj_seq_dict = sequence_mod_data["sj_sequence"]
        self.sj_inds_df = sequence_mod_data["sj_inds_df"]
        self.max_prime_seq_len = sequence_mod_data["max_prime_seq_len"]
        self.max_sj_seq_len = sequence_mod_data["max_sj_seq_len"]
        self.pad_indx = sequence_mod_data["pad_indx"]
        self.events_coordinates_path = sequence_mod_data["events_coordinates_path"]
        # self.gene_embed_dict = gene_embed_dict 
        self.NEURON_TYPE_ENCODING = neuron_type_fn()

    def setup(self):
        filtered_df = self.data_csv[self.data_csv["p_seq_len"] < self.max_prime_seq_len]
        self.filtered_df = filtered_df
        if 'neuron_replicate' not in self.dataset_type:
            if not self.events_coordinates_path:
                raise ValueError(
                    "events_coordinates is required for this dataset_type "
                    f"({self.dataset_type!r}) but none was provided. Add "
                    "[files] events_coordinates to data_config.ini. "
                    "(Not needed for neuron_replicate data.)"
                )
            self.events_coordinates = pd.read_csv(self.events_coordinates_path, sep=None, engine='python')
            # O(1) lookup by event_id (one row per event). Avoids O(N) pandas
            # filters per __getitem__ call.
            self.events_coords_by_event = {
                r['event_id']: r for r in self.events_coordinates.to_dict('records')
            }
        # sj_inds_df is also unique per event_id.
        self.sj_inds_by_event = {
            r['event_id']: r for r in self.sj_inds_df.to_dict('records')
        }
        # GTEx and worm differ on prime_seq_dict keying:
        #   - GTEx: per-event 25kb windows, keyed by event_id
        #   - worm: per-gene full sequences, keyed by gene_id (WBGene*)
        # Pick the mode once at setup so __getitem__ doesn't have to.
        sample = filtered_df.iloc[0]
        if sample["event_id"] in self.prime_seq_dict:
            self._prime_seq_key = "event_id"
        elif sample["gene_id"] in self.prime_seq_dict:
            self._prime_seq_key = "gene_id"
        else:
            raise KeyError(
                f"prime_seq_dict has neither event_id {sample['event_id']!r} nor "
                f"gene_id {sample['gene_id']!r} for the first row; check enc_seq_file path."
            )
     
    def __getitem__(self, idx):
        raw_datapoint = self.filtered_df.iloc[idx]  

    def __getitem__(self, idx):
        raw_datapoint = self.filtered_df.iloc[idx]

        neuron_idx = torch.LongTensor([self.NEURON_TYPE_ENCODING[raw_datapoint["neuron"]]]).reshape(1, -1)

        if 'neuron_replicate' in self.dataset_type:
            meta_data = {
                'gene_id': raw_datapoint["gene_id"],
                'event_id': raw_datapoint["event_id"],
                'nb_reads': raw_datapoint["nb_reads"],
                'neuron': raw_datapoint["neuron"],
                'neuron_replicate': raw_datapoint['neuron_replicate'],
                'p_seq_len': raw_datapoint['p_seq_len'],
                'sj_seq_len': raw_datapoint['sj_seq_len'],
                'exon_start': raw_datapoint['exon_start'],
                'exon_end': raw_datapoint['exon_end'],
                'intron_start': raw_datapoint['intron_start'],
                'intron_end': raw_datapoint['intron_end'],
                'gene_length': raw_datapoint['gene_length'],
                'neuron_idx': neuron_idx
            }
        else: 
            events_coordinates_i = self.events_coords_by_event[raw_datapoint['event_id']]
            exon_start = events_coordinates_i['exon_start']
            exon_end = events_coordinates_i['exon_end']
            intron_start = events_coordinates_i['intron_start']
            intron_end = events_coordinates_i['intron_end']
            gene_length = events_coordinates_i['gene_length']
            meta_data = {
                'gene_id': raw_datapoint["gene_id"],
                'event_id': raw_datapoint["event_id"],
                'nb_reads': raw_datapoint["nb_reads"],
                'neuron': raw_datapoint["neuron"],
                'p_seq_len': raw_datapoint['p_seq_len'],
                'sj_seq_len': raw_datapoint['sj_seq_len'],
                'gene_length': gene_length,
                'exon_start': exon_start,
                'exon_end': exon_end,
                'intron_start': intron_start,
                'intron_end': intron_end,
            }
  
        # neuron type
        # sequence modality
        p_seq = self.prime_seq_dict[raw_datapoint[self._prime_seq_key]]
        sj_data = self.sj_seq_dict[raw_datapoint["event_id"]]

        # construct sequence
        p_seq = p_seq.reshape(1, -1)
        sj_annot_seq = sj_data[1, :].reshape(1, -1)

        # get splice region coordinates (O(1) dict lookup)
        sj_row = self.sj_inds_by_event[raw_datapoint["event_id"]]
        sj_st_indx = int(sj_row["sj_st_indx"])
        sj_end_indx = int(sj_row["sj_end_indx"])
        end_indx = len(p_seq.flatten())

        # # append to meta data
        # seq_data.append((sj_st_indx, sj_end_indx))

        # convert intron notation to whole sequence notation
        sj_annot_seq[sj_annot_seq == 1] = 1  # exon = 1 (redundant but clarifies next line)
        sj_annot_seq[sj_annot_seq == 0] = 2  # intron = 2

        # flanking rna
        left_p_seq_pad = sj_st_indx
        right_p_seq_pad = end_indx - sj_end_indx
        padded_p_annot_seq = F.pad(sj_annot_seq, (left_p_seq_pad, right_p_seq_pad), value=3)

        # sequence padding
        p_seq = p_seq.flatten()
        p_padding_size = self.max_prime_seq_len - len(p_seq.flatten())
        padded_p_seq = F.pad(p_seq, (0, p_padding_size), value=self.pad_indx)
        padded_p_seq = padded_p_seq.reshape(1, -1)

        # annotation padding
        padded_p_annot_seq = padded_p_annot_seq.flatten()
        padded_p_annot_seq = F.pad(padded_p_annot_seq, (0, p_padding_size), value=self.pad_indx)
        padded_p_annot_seq = padded_p_annot_seq.reshape(1, -1)

        # Edge events: ~0.4% of GTEx events have sj_end_indx - sj_st_indx < 500
        # (cassette exon near window boundary), which makes the F.pad above overshoot
        # max_prime_seq_len. Truncate to guarantee uniform shape for batch collate.
        padded_p_seq       = padded_p_seq[..., :self.max_prime_seq_len]
        padded_p_annot_seq = padded_p_annot_seq[..., :self.max_prime_seq_len]
        # target 
        psi_val = torch.Tensor([raw_datapoint["PSI"]]).reshape(1, 1)

        nb_reads = torch.Tensor([raw_datapoint["nb_reads"]]).reshape(1, 1)

        if 'singlereplicant' in self.dataset_type:
            delta_psi_val = torch.Tensor([raw_datapoint["DELTA PSI"]]).reshape(1, 1)
            targets = {
                'psi': psi_val,
                'delta_psi': delta_psi_val,
            }
        else:
            targets = {
                'psi': psi_val,
            }

        processed_datapoint = [
            meta_data,
            [
                padded_p_seq,
                padded_p_annot_seq,
                neuron_idx
            ],
            targets
        ]

        return processed_datapoint

    def __len__(self):
        return len(self.filtered_df)


def neuron_type_fn():
    NEURON_TYPE_ENCODING = {
        "AVH": 0,
        "CAN": 1,
        "VD": 2,
        "AFD": 3,
        "RIM": 4,
        "AWC": 5,
        "AIY": 6,
        "RIC": 7,
        "LUA": 8,
        "AVE": 9,
        "RIS": 10,
        "PVC": 11,
        "ADL": 12,
        "ASK": 13,
        "DD": 14,
        "PVM": 15,
        "ASG": 16,
        "AVA": 17,
        "NSM": 18,
        "AVK": 19,
        "ASER": 20,
        "OLQ": 21,
        "SMD": 22,
        "DA": 23,
        "AVM": 24,
        "DVC": 25,
        "VC": 26,
        "AIN": 27,
        "OLL": 28,
        "ASI": 29,
        "IL1": 30,
        "IL2": 31,
        "AIM": 32,
        "SMB": 33,
        "RMD": 34,
        "BAG": 35,
        "AWB": 36,
        "AVL": 37,
        "AWA": 38,
        "AVG": 39,
        "ASEL": 40,
        "VB": 41,
        "PHA": 42,
        "PVD": 43,
        "RIA": 44,
        "I5": 45,
        "CEP":46,
        "PVP":47,
        "PVQ":48,
        "DVB":49,
        "HSN":50,
        "RME":51,
        "SIA":52,
        # "PVPx":53,
        # "PVPx":54,
        # "PVPx":55,
        # "PVPx":56,
    }
    try:
        from data.gtex_tissue_encoding import gtex_tissue_type_encoding

        NEURON_TYPE_ENCODING.update(
            gtex_tissue_type_encoding(len(NEURON_TYPE_ENCODING))
        )
    except (FileNotFoundError, ImportError):
        pass
    return NEURON_TYPE_ENCODING
