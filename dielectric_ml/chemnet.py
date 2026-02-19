# chemnet.py; chemical information tools for neural network

from tqdm import tqdm

import numpy as np
from sklearn.preprocessing import StandardScaler
import skfp.fingerprints as fp # wonderful fingerprinting tool.

from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import AllChem


'''
list of functions:
    
    scale_data - scales some data
    generate_low_energy_conformers - generates low energy conformers with ETKDGv3
    get_all_fingerprints - interface to skfp fingerprinting tool.
    filter_single_fingerprint_and_targets - filtering of fingerprints and targets (temperatures) to remove null entries
'''

def scale_data(data):
    '''
    simple script to scale/normalize data
    '''
    print(f'Scaling data with shape {np.shape(data)}')
    return StandardScaler().fit_transform(data)


def _is_valid_smiles(x) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0


def _has_conf_id(m) -> bool:
    return isinstance(m, Mol) and m.HasProp("conf_id")


def filter_valid_3d_rows(df, conformer_column: str = "min_e_conf"):
    """
    Return a filtered COPY of df where df[conformer_column] is an RDKit Mol
    with 'conf_id' property set.
    """
    mask = df[conformer_column].apply(_has_conf_id)
    return df.loc[mask].reset_index(drop=True)


def defined_fp_methods(use_3D = False, n_jobs = 1, fp_size = 1024):
    '''
    This is just a cheat function for getting the list of fingerprints in skfp.fingerprints
    
    For fp types that can have 3D coordinates we'll define them seperately (with "_3D" at the end)
    
    The list was manually curated some time ago. Seemingly the code now has some extra fingeprint 
    types in it, but I haven't updated this (yet)
    
    https://scikit-fingerprints.readthedocs.io/latest/examples/02_fingerprint_types.html
    https://scikit-fingerprints.readthedocs.io/latest/modules/fingerprints.html
    
    you can:
    -   toggle 3d on/off (use_3D = False/True); 
    -   set the number of cores to use (n_jobs - note, this seems to affect the results... jus leave it at 1)
    -   change the bit depth (fp_size; some integer).
    '''
    fp_methods = {
        "atom_pair_fingerprint": fp.AtomPairFingerprint(n_jobs=n_jobs, fp_size=fp_size),
        "autocorr_fingerprint": fp.AutocorrFingerprint(n_jobs=n_jobs),
        "ecfp_fingerprint": fp.ECFPFingerprint(n_jobs=n_jobs, fp_size=fp_size),
        "erg_fingerprint": fp.ERGFingerprint(n_jobs=n_jobs),
        "estate_fingerprint": fp.EStateFingerprint(n_jobs=n_jobs),
        "ghose_crippen_fingerprint": fp.GhoseCrippenFingerprint(n_jobs=n_jobs),
        "klekotha_roth_fingerprint": fp.KlekotaRothFingerprint(n_jobs=n_jobs),
        "laggner_fingerprint": fp.LaggnerFingerprint(n_jobs=n_jobs),
        "layered_fingerprint": fp.LayeredFingerprint(n_jobs=n_jobs, fp_size=fp_size),
        "lingo_fingerprint": fp.LingoFingerprint(n_jobs=n_jobs),
        "maccs_fingerprint": fp.MACCSFingerprint(n_jobs=n_jobs),
        "mapf_fingerprint": fp.MAPFingerprint(n_jobs=n_jobs, fp_size=fp_size),
        "mordred_fingerprint": fp.MordredFingerprint(n_jobs=n_jobs),
        "mqns_fingerprint": fp.MQNsFingerprint(n_jobs=n_jobs),
        "pattern_fingerprint": fp.PatternFingerprint(n_jobs=n_jobs, fp_size=fp_size),
        #"pharamacophore_fingerprint": fp.PharmacophoreFingerprint(n_jobs=n_jobs, fp_size=fp_size),
        "pubchem_fingerprint": fp.PubChemFingerprint(n_jobs=n_jobs),
        "rdkit_fingerprint": fp.RDKitFingerprint(n_jobs=n_jobs, fp_size=fp_size),
        "topo_tors_fingerprint": fp.TopologicalTorsionFingerprint(n_jobs=n_jobs, fp_size=fp_size),
        }
    if use_3D:
        if n_jobs != 1:
            print("\nWarning: overriding n_jobs=1 for 3D descriptors.")
        n_jobs_3d = 1
        fp_methods.update({
            "atom_pair_fingerprint_3D": fp.AtomPairFingerprint(n_jobs=n_jobs_3d, fp_size=fp_size, use_3D=True),
            "autocorr_fingerprint_3D": fp.AutocorrFingerprint(n_jobs=n_jobs_3d, use_3D=True),
            "e3fp_fingerprint_3D": fp.E3FPFingerprint(n_jobs=n_jobs_3d, fp_size=fp_size),
            "getaway_fingerprint_3D": fp.GETAWAYFingerprint(n_jobs=n_jobs_3d),
            "mordred_fingerprint_3D": fp.MordredFingerprint(n_jobs=n_jobs_3d, use_3D=True),
            "morse_fingerprint_3D": fp.MORSEFingerprint(n_jobs=n_jobs_3d),
            #"pharamacophore_fingerprint_3D": fp.PharmacophoreFingerprint(n_jobs=n_jobs_3d, fp_size=fp_size),
            "rdf_fingerprint_3D": fp.RDFFingerprint(n_jobs=n_jobs_3d),
            "usr_fingerprint_3D": fp.USRFingerprint(n_jobs=n_jobs_3d),
            "usrcat_fingerprint_3D": fp.USRCATFingerprint(n_jobs=n_jobs_3d),
            "whim_fingerprint_3D": fp.WHIMFingerprint(n_jobs=n_jobs_3d),
        })
    return fp_methods
    
def get_fingerprints_new(df,
    smiles_column: str = "SMILES",
    conformer_column: str = "min_e_conf",
    n_jobs: int = 1,
    fp_size: int = 1024,
    use_3D: bool = False,
    fp_type: str | None = None,
    print_output: bool = False,
    drop_invalid_3d: bool = True):

    from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer
    from tqdm import tqdm

    mol_from_smiles = MolFromSmilesTransformer()
    mols_list = mol_from_smiles.transform(list(df[smiles_column]))

    fp_methods = defined_fp_methods(use_3D=use_3D, n_jobs=n_jobs, fp_size=fp_size)

    if use_3D:
        conf_gen = ConformerGenerator()
        mols_list = conf_gen.transform(mols_list)

        # --- sanity: ensure conformers actually exist ---
        has_conf = np.array([(m is not None and m.GetNumConformers() > 0) for m in mols_list], dtype=bool)

        if drop_invalid_3d:
            mols_list = [m for m, ok in zip(mols_list, has_conf) if ok]
        elif not np.all(has_conf):
            bad = int((~has_conf).sum())
            raise ValueError(f"{bad} molecules have no conformers after ConformerGenerator(); set drop_invalid_3d=True or fix conformer generation.")

        # --- sanity: ensure we didn't accidentally select only 2D methods ---
        # (requires your defined_fp_methods to tag 3D ones; if not available, at least warn)
        if fp_type is None:
            maybe_3d = [k for k in fp_methods.keys() if "3d" in k.lower() or "3D" in k]
            if len(maybe_3d) == 0:
                print("WARNING: use_3D=True but no fingerprint names look 3D. You may be computing 2D fingerprints on 3D-embedded mols.")

    fingerprints = {}
    print("Generating Fingerprints...")
    if fp_type:
        if fp_type not in fp_methods:
            raise ValueError(f"Invalid fp_type: {fp_type}. Available types: {list(fp_methods.keys())}")
        fingerprints[fp_type] = fp_methods[fp_type].transform(mols_list)
    else:
        for fp_name, fp_method in tqdm(fp_methods.items()):
            fingerprints[fp_name] = fp_method.transform(mols_list)

    print(f"Fingerprint generation complete; generated {len(fingerprints)} types of fingerprints")
    return fingerprints

    
def convert_fingerprints_to_array(fingerprints):
    """
    because of the way the skfp fingerprints work we are storing the 3D ones as lists
    which causes an issue later on. So we'll use this to convert them to a np.array
    """
    for key, value in fingerprints.items():
        if isinstance(value, list):
            fingerprints[key] = np.array(value)
    return fingerprints
    
def clean_fprints_targets(fingerprint_list, target_values, remove_zero_targets=False, add_bool_fingerprint = False, print_output = False):
    """
    Filters out entries where the fingerprint is None, and optionally where the target value is zero.
    NaN values in fingerprints are replaced with zero. If remove_zero_targets is False, generates boolean values 
    on the fly to indicate whether the target is zero, and appends these to the fingerprints.

    args:
        fingerprint_list    - list of fingerprints for a single fingerprint type.
        target_values       - target values corresponding to the fingerprints (np.array; probably).
        remove_zero_targets - if True, filter out entries where the target value is zero. If False, append a 
                                    boolean indicator to the fingerprints indicating whether the target is zero.

    Returns:
        filtered_fingerprints   -  list with None entries removed and NaNs replaced with zeros, potentially 
                    with an appended boolean indicator if remove_zero_targets is False.
        filtered_targets        - filtered target values with entries corresponding to None fingerprints removed.
    """

    filtered_fingerprints = [] # use these lists to store the filtered data
    filtered_targets = []
    
    if add_bool_fingerprint and print_output:
        print('Adding additional boolean fingerprint for transition presence (=1) or absence (=0)...')  

    for i in range(len(fingerprint_list)):
        fingerprint = fingerprint_list[i]
        target = target_values[i]
        
        if type(target) != str:
            if np.isnan(target) == False:
                if fingerprint is not None:
                    fingerprint = np.nan_to_num(fingerprint)  # Replace NaNs with zeros                
                    
                    if not remove_zero_targets and add_bool_fingerprint: # if not, we'll get boolean value on the fly (1 if target != 0, else 0)
                        boolean_value = 1 if target != 0 else 0
                        fingerprint = np.append(fingerprint, boolean_value) # Append the boolean value directly to the end of the fingerprint array

                    if not remove_zero_targets or target != 0:
                        filtered_fingerprints.append(fingerprint)
                        filtered_targets.append(target)

    filtered_fingerprints = np.array(filtered_fingerprints)
    filtered_targets = np.array(filtered_targets)
    
    return filtered_fingerprints, filtered_targets