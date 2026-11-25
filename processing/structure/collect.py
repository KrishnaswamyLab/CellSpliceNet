"""Combine structure_dict.pkl with the scattering moments -> <data_root>/<organism>/processed/structure/structure_scattering_dict.pkl (structure step 3)."""
import argparse
import pickle as pk
import tqdm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from processing.common import add_location_args, work_dir_for  # noqa: E402

def main():
    parser = argparse.ArgumentParser(description='Combine structure dictionary with scattering moments.')
    add_location_args(parser)
    args = parser.parse_args()

    base_dir = work_dir_for(args)

    with open(f'{base_dir}/structure_dict.pkl', 'rb') as f:
        structure_dictionary = pk.load(f)

    structure_dictionary['structure_scattering_coeffs'] = []

    with open(f'{base_dir}/structure_scattering_moments.pkl', 'rb') as scatter_f:
        for _ in tqdm.tqdm(range(len(structure_dictionary['structure_seq']))):
            structure_dictionary['structure_scattering_coeffs'].append(pk.load(scatter_f))


    # final location, or the work folder for test runs (--work-dir)
    out = Path(base_dir) if args.work_dir else Path(args.data_root) / args.organism / 'processed' / 'structure'
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'structure_scattering_dict.pkl', 'wb') as f:
        pk.dump(structure_dictionary, f)
    print(f'wrote {out / "structure_scattering_dict.pkl"} ({len(structure_dictionary["event_id"])} events)')


if __name__ == "__main__":
    main()
