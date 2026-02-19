"""
Processing data so that it can be used for training models 
"""
import torch
from torch_geometric.data import Data

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

import re

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

def process_exp_data(file_path, prop_col=None, transition=None, preserve_columns=None):
    """
    process experimental data from a spreadsheet
    
    
    returns: pandas dataframe 

    args: 
        file_path: path to experimental data
        prop_col: the title of the column where the property is found
        transition: type of transition to find temperature of
        preserve_columns: list of column names to preserve from original Excel
                         (e.g., ['title', 'CAS', 'Reference'])
    """
    # load exp data
    df = pd.read_excel(file_path) 
    # Drop rows where SMILES or Transition temperatures are NaN
    df = df.dropna(subset=["smiles"])
    
    # Store original index for tracking
    original_indices = df.index.tolist()
    
    # Extract SMILES and transition temperatures
    smiles_series = df["smiles"]  # Assuming "SMILES" is the column name

    
    
    if transition is not None:
        # temps series is a formatted list of transitions
        temps_series = df["Transitions / dC"]  # Assuming "Transitions / dC" is the correct column
    else:
        temps_series = pd.Series([None] * len(smiles_series), index=df.index)

    if prop_col is not None:
        prop_series = df[prop_col]
    else:
        prop_series = pd.Series([None] * len(smiles_series), index=df.index)
    
    # Handle preserved columns - default to 'title' if it exists
    if preserve_columns is None:
        preserve_columns = []
    
    # Always try to preserve 'title' if it exists and not already specified
    if 'title' in df.columns and 'title' not in preserve_columns:
        preserve_columns = ['title'] + list(preserve_columns)
    
    # Extract preserved column data
    preserved_data = {}
    for col in preserve_columns:
        if col in df.columns:
            preserved_data[col] = df[col].tolist()
        else:
            print(f"Warning: Column '{col}' not found in Excel file")
    
    # store values for dataframe
    smiles = []
    rdkit_molecules = []
    mol_sizes = []
    transition_temps = []
    props = []
    phase_presences = []
    kept_indices = []  # Track which rows we keep
    preserved_values = {col: [] for col in preserved_data.keys()}
    
    for idx, (smi, prop, temp) in enumerate(tqdm(
            zip(smiles_series, prop_series, temps_series), 
            desc="Processing molecules",
            total=len(smiles_series))):
        



        try:
            # Initialize molecule object and convert SMILES to RDKit Mol object
            mol = Molecule(smi, temp)  
            rdkit_mol = Chem.MolFromSmiles(mol.smiles)
            
            try: 
                prop = float(prop)
            except(ValueError, TypeError):
                prop = np.nan



            # Get the transition temperature for the specified phase (e.g., "I")
            transition_temp = mol.get_transition_temp(transition)

            # check for the presence of the phase by seeing if gtt returns a temp
            phase_temp = mol.get_transition_temp(transition)
            if phase_temp is np.nan:
                phase_presence = 0
            else:
                phase_presence = 1

            # mol sizes for vae
            if rdkit_mol is not None:
                mol_size = len(rdkit_mol.GetAtoms())
            else:
                mol_size = 0
  
            # Append valid molecules and data
            transition_temps.append(transition_temp)  
            rdkit_molecules.append(rdkit_mol)  
            smiles.append(mol.smiles)
            mol_sizes.append(mol_size)
            phase_presences.append(phase_presence)
            props.append(prop)
            kept_indices.append(original_indices[idx])
            
            # Preserve additional column values
            for col, values in preserved_data.items():
                preserved_values[col].append(values[idx])
        
        # print error if smiles processing fails
        except Exception as e:
            print(f"Error processing molecule {smi}: {e}")

    print(f"test: smiles and temps equal: {len(smiles), len(transition_temps)}")

    # Build output dataframe
    output_data = {
        "SMILES": smiles,
        "LENGTH": mol_sizes,
        "TRANS_TEMP": transition_temps,
        "RDKIT_MOL": rdkit_molecules,
        "PHASE_PRESENCE": phase_presences,
        "PROP": props,
        "ORIGINAL_INDEX": kept_indices,  # Track original Excel row
    }
    
    # Add preserved columns
    for col, values in preserved_values.items():
        output_data[col] = values
    
    processed_df = pd.DataFrame(output_data)
    return processed_df

def coders(molecules:list):
    """
    create atom and bond encoders/decoders from list of molecules for VAE

    args:
        molecules: list of molecules to get the coders for
    """
    atom_labels = sorted(set([atom.GetAtomicNum() for mol in molecules for atom in mol.GetAtoms()] + [0]))
    
    # encode atom type (idx:atom_num)
    atom_encoder = {l: i for i, l in enumerate(atom_labels)}
    atom_decoder = {i: l for i, l in enumerate(atom_labels)}

    bond_labels = [Chem.rdchem.BondType.ZERO] + list(sorted(set(bond.GetBondType() for mol in molecules for bond in mol.GetBonds())))

    # encode bond type (idx:bond type) 
    bond_encoder = {l: i for i, l in enumerate(bond_labels)}
    bond_decoder = {i: l for i, l in enumerate(bond_labels)}

    return atom_encoder, atom_decoder, bond_encoder, bond_decoder 

def graph_features(mol, atom_labels, max_length=None):
    """
    creates a one hot encoded node matrix for a rdkit molecule 

    args:
        mol: rdkit molecule object
        atom_labels: list of atomic numbers for atoms present
        max_length: max_length of molecules to be used in script
    """
    # set max length if not already set
    max_length = max_length if max_length is not None else mol.GetNumAtoms()
    
    # one hot encoded nodes for molecule 
    features = np.array([[*[a.GetAtomicNum() == i for i in atom_labels]] for a in mol.GetAtoms()], dtype=np.int32)

    # vpstack adds zero vectors if shape is less than max lengths
    return np.vstack((features, np.zeros((max_length - features.shape[0], features.shape[1]))))

def feature_size(mol, atom_labels, max_length=None): 
    """
    create one hot encoded feature tensor

    args:
        mol: rdkit molecule object
        atom_labels: list of atomic numbers for atoms present
        max_lenght: max length of molecules to be used in script
    """

    # node features matrix to torch tensor
    feature = graph_features(mol, atom_labels, max_length)
    feature = torch.cat([torch.tensor(feature), torch.zeros([max_length-feature.shape[0], feature.shape[1]])], 0)
    
    # one hot encode no atom to first column in no matrix
    for i in range(feature.shape[0]):
        if 1 not in feature[i]:
            feature[i, 0] = 1
    
    # return feature tensor
    return feature

def graph_adjacency(mol, atom_number, bond_encoder_m, connected=True):
    """
    using the bond encoder it creates a bond tensor for each node in max_length
    returns tensor of shape (max_length, bond_types * max_length) 

    args:
        mol: r
        atom_number: max_length
        bond_encoder_m:  
        connected:
    """
    # 0 matrix of shape (max_length, max_length) 
    A = np.zeros(shape=(atom_number, atom_number), dtype=np.int32)
    
    # indices of where bonds begin and end
    begin, end = [b.GetBeginAtomIdx() for b in mol.GetBonds()], [b.GetEndAtomIdx() for b in mol.GetBonds()]
    
    # list of bond types in molecule
    bond_type = [bond_encoder_m[b.GetBondType()] for b in mol.GetBonds()]
    
    # create adjacency matrix by assigning bond types to begin end indices
    A[begin, end] = bond_type
    A[end, begin] = bond_type

    # number of bonds for each atom from the adjacency matrix 
    degree = np.sum(A[:mol.GetNumAtoms(), :mol.GetNumAtoms()], axis=-1)
    
    # assign adj
    adj = A if connected and (degree > 0).all() else None
    
    # remove lower triangle of adj matrix
    for i in range(adj.shape[0]):
        adj[i, 0:i] = 0
    
    # create a list of one hot encoded bonds adjacency matrix for each node
    oh_list = []
    for i in range(adj.shape[0]):
        oh = np.zeros(shape=(atom_number, 5), dtype=np.int32)
        for j in range(adj.shape[1]):
            oh[j, adj[i][j]] = 1
        oh_list.append(torch.tensor(oh))

    # for element in oh list turn it into 1d tensor
    return torch.cat([o for o in oh_list], 1)

def graph2mol(node_labels, adjacency, atom_decoder_m, bond_decoder_m, strict=True):
    mol = Chem.RWMol()
    for node_label in node_labels:
        mol.AddAtom(Chem.Atom(atom_decoder_m[node_label]))
    for start, end in zip(*np.nonzero(adjacency)):
        if start < end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adjacency[start, end]])
    if strict:
        try:
            Chem.SanitizeMol(mol)
        except:
            mol = None
    return mol

def one_hot_encoding(x, permitted_list):
    """
    maps input elements x which are not in the permitted list to the last element of the permitted list
    """

    if x not in permitted_list:
        x = permitted_list[-1]

    # what does this do 
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding

def get_atom_features(atom, use_chirality = True, hydrogens_implicit = True):
    """
    Takes an RDKit atom object as an input and gives a 1D numpy array of atom features as an output
    """
    
    # define list of permitted atoms 
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na',
                                'Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb',
                                'Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd',
                                'In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

    # compute atom features
    # one hot encode atom 
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)

    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])

    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])

    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2",
    "SP3", "SP3D", "SP3D2", "OTHER"])

    is_in_a_ring_enc = [int(atom.IsInRing())]

    is_aromatic_enc = [int(atom.GetIsAromatic())]

    atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.029)]

    # scaled using emperically estimated properties
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())) - 1.5 / 0.6)]

    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())) - 0.64 / 0.76)]

    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled

    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc

    if hydrogens_implicit == True: 
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond):
    """
    Get bond features as a numerical array for a given rdkit bond object.
    """
    # Bond type one-hot
    bond_types = [Chem.rdchem.BondType.SINGLE,
                  Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE,
                  Chem.rdchem.BondType.AROMATIC]
    bond_type = [0] * len(bond_types)
    bt = bond.GetBondType()
    if bt in bond_types:
        bond_type[bond_types.index(bt)] = 1
    
    # Conjugation
    conjugated = [1 if bond.GetIsConjugated() else 0]
    
    # In ring
    in_ring = [1 if bond.IsInRing() else 0]
    
    # Stereo
    stereo_types = [Chem.rdchem.BondStereo.STEREONONE,
                    Chem.rdchem.BondStereo.STEREOZ,
                    Chem.rdchem.BondStereo.STEREOE]
    stereo = [0] * len(stereo_types)
    st = bond.GetStereo()
    if st in stereo_types:
        stereo[stereo_types.index(st)] = 1
    
    return bond_type + conjugated + in_ring + stereo

def create_pytorch_geometric_graph_data_list_from_smiles(x_smiles):
    """
    inputs: 
    x_smiles = list of smiles strings
    y = list of numerical labels for the smiles strings

    outputs:
    data_list: a list of torch_geometric.data.Data objects which represent labelled molecular graphs
    """

    data_list = []

    for smiles in x_smiles:
        # convert smiles to rdkit mol object
        mol = Chem.MolFromSmiles(smiles)

        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))

        # construct node feature matrix x of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))

        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)

        X = torch.tensor(X, dtype = torch.float)

        # construct edge index array
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)

        # construct edge features array of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))
        
        for (k, (i, j)) in enumerate(zip(rows, cols)):
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))

        EF = torch.tensor(EF, dtype = torch.float)

        # construct label tensor 
        data_list.append(Data(x= X, edge_index= E, edge_attr = EF, smiles = smiles))

    return data_list

def classifier_dataloader(df: pd.DataFrame, prop_col: str):
    """
    create graph dataloaders with nf and temp labels 

    args:
        df: processed dataframe of experimental data
        prop_col: title of the column in which the binary values 

    return data_list for k fold stuff 
    """
      
    # turn source   df smiles into graph data
    data_list = create_pytorch_geometric_graph_data_list_from_smiles(df["SMILES"].to_list())

    prop_list = df[prop_col].to_list()

    # binary values
    for i in range(len(data_list)):
        prop = prop_list[i]
        prop = torch.tensor(prop , dtype=torch.float32).reshape(-1, 1)
        data_list[i].prop = prop
    
    # float for datatype reasons
    for i in range(len(data_list)):
        data_list[i].x = data_list[i].x.float()

    return  data_list

def predictor_dataloader(df: pd.DataFrame, prop_col: str):
    """
    Create the dataloaders for predictor tasks
    """

    scaler_y = StandardScaler()
    
    # turn source   df smiles into graph data
    data_list = create_pytorch_geometric_graph_data_list_from_smiles(df["SMILES"].to_list())

    # what is unnormalised continuous property 
    for i in range(len(data_list)):
        data_list[i].prop = df[prop_col].to_list()[i]

    # float for datatype reasons
    for i in range(len(data_list)):
        data_list[i].x = data_list[i].x.float()

    # scaling temperatures 
    prop_list = df[prop_col].to_list()
    norm_prop = scaler_y.fit_transform(np.array(prop_list).reshape(-1,1))

    # add scaled property to data_list
    for i in range(len(data_list)):
        norm_prop_tensor =  torch.tensor(norm_prop[i], dtype=torch.float32).reshape(-1,1)
        data_list[i].norm_prop = norm_prop_tensor

    return  data_list, scaler_y

class Molecule:
    def __init__(self, smiles, transitions):
        """
        Initialize the molecule with a SMILES string and transition temperatures.
        """
        self.smiles = smiles
        self.transitions = transitions
        self.transition_dict = {}
        self.low_energy_conformer = None

    
    def gen_low_energy_conformer(self, num_conformers=3, max_steps=1000, threshold=1e-4):
        """
        Generate the lowest energy conformer using the MMFF94 force field.

        Args:
            num_conformers (int): Number of conformers to generate.
            max_steps (int): Maximum optimization steps.
            threshold (float): Energy convergence threshold.

        Returns:
            mol_low_energy: Molecule with the lowest energy conformer or None if failed.
        """
        mol = Chem.AddHs(Chem.MolFromSmiles(self.smiles))
            
        if mol is None:
            print(f"Invalid SMILES string: {self.smiles}")
            return None
        
        # Generate conformers using ETKDGv3
        params = AllChem.ETKDGv3()
        conformer_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)

        if not conformer_ids:
            print(f"Conformer embedding failed for {self.smiles}")
            return None

        energies = []
        for conf_id in conformer_ids: 
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
            if mmff_props is None: 
                print(f"MMFF cannot be used for molecule: {self.smiles}")
                return None

            try:
                # Create force field and minimize
                ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
                if ff is not None:
                    ff.Minimize(maxIts=max_steps, forceTol=threshold)
                    energy = ff.CalcEnergy()
                    energies.append((conf_id, energy))
            except Exception as e:
                print (f"MMFF optimisation error for {self.smiles}: {str(e)}")
                return None

        if energies:
            # Sort conformers by energy
            energies.sort(key=lambda x: x[1])
            lowest_energy_conf_id = energies[0][0]

            # create new molecule with lowest energy conformer
            mol_low_energy = Chem.Mol(mol)
            mol_low_energy.RemoveAllConformers()
            lowest_conf = mol.GetConformer(lowest_energy_conf_id)
            mol_low_energy.AddConformer(lowest_conf, assignId=True)

            mol_low_energy.SetIntProp("conf_id", 0)
            self.low_energy_conformer = mol_low_energy

            return mol_low_energy

        else:
            print(f"No valid conformers found for {self.smiles}")
            return None
            
    def get_transition_temp(self, transition_of_interest):
        # Extract SMILES and temperature transition columns  
        parsed_data = []
        final_temps = []
        
        # Compile regex patterns once for efficiency
        phase_pattern = re.compile(r'[A-Za-z]+')
        temp_pattern = re.compile(r'[\d.]+')
        
        if self.transitions is not None:
            trans_str = str(self.transitions)  # Ensure the transition data is a string
            phases = phase_pattern.findall(trans_str)
            temp_values = temp_pattern.findall(trans_str)
            #print(phases, temp_values)
            
            # Convert temperature values to Kelvin and round
            temps_k = [np.round(float(temp) + 273.15, 1) for temp in temp_values]
            
            trans_dict = {}
            
            # Handle the case where no temp after K
            if phases and phases[0] == 'K':
                if not temps_k or not re.match(r'[\d.]+', trans_str.split('K', 1)[1].strip()):
                    trans_dict[phases[0]] = np.nan
                    phases = phases[1:]  # remove k
            
            # Create the transition dictionary
            # Assign temperature values that follow each phase letter to that phase
            for i in range(len(phases)):
                if i < len(temps_k):
                    # Assign the temperature that follows this phase letter to this phase
                    trans_dict[phases[i]] = temps_k[i]
                elif i == len(phases) - 1 and len(temps_k) > 0:
                    # For the last phase when no temperature follows, use the previous temperature
                    trans_dict[phases[i]] = temps_k[-1]
            
            parsed_data.append(trans_dict)
        else:
            parsed_data.append({})
        
        # Extract temperatures corresponding to the specified transition
        for trans_dict in parsed_data:
            final_temp = np.nan  # Default to NaN if no matching transitions found
            for key in trans_dict:
                # Check for exact match only (case-insensitive)
                if key.lower() == transition_of_interest.lower():
                    final_temp = trans_dict[key]
                    break
        
        self.transitions_dict = parsed_data
        
        return final_temp
