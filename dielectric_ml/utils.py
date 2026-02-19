"""
* saving the different models
* tracking experiment functionality 

"""

import torch
from pathlib import Path
import os
from dielectric_ml import models
import pickle
from datetime import datetime
import re
import pandas as pd
import glob
from rdkit import Chem
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from rdkit.Chem import Draw, rdDepictor


def save_model(model: torch.nn.Module,
               target_dir: str = "models",
               model_name: str = "last_run.pth"
               ):
    """
    saves model to models directory
    """
    # create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with .pt or .pth"
    model_save_path = target_dir_path / model_name

    print(f"saving model to {model_save_path}")
    torch.save(obj=model.state_dict(), f = model_save_path)

def load_params(model: torch.nn.Module,
               target_dir: str = "models",
               model_name: str = "last_run.pth",
               ):
    """
    helper function for loading model parameters (for inference only)
    """
    
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with .pt or .pth"
    model_load_path = target_dir_path / model_name
    
    model = model.load_state_dict(torch.load(model_load_path, weights_only=True, map_location=torch.device('cpu')))
    return model

def save_generated_mols(gen_df:pd.DataFrame, source_df = None, remove_duplicates: bool = True, file_name = "generated_molecules.xlsx"):
    """
    save the generated mols dataframe to an excel file with the option of removing duplicate smiles strings 
    """
    target_dir = Path("data")
    target_dir.mkdir(exist_ok=True)
    
    file_path = target_dir / file_name
    
    # get generated smiles 
    gen_df_list = gen_df["SMILES"].to_list()
    canon_gen_smiles = []  
    # check list of generated mols not empty
    if gen_df_list:
        for x in gen_df_list:
            try: 
                gen_mol = Chem.MolFromSmiles(x)
                canon_gen_smiles.append(Chem.MolToSmiles(gen_mol))
            except:
                print(f"smiles_error: {x}")

    # get source list of smiles in case of remove duplicated

    if source_df is not None:
        source_smiles = source_df["SMILES"].to_list()
        canon_source_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in source_smiles]

    if remove_duplicates:
        set1, set2 = set(canon_gen_smiles), set(canon_source_smiles)

        common_elements = set1.intersection(set2)

        gen_df_no_dupes_list = [item for item in gen_df_list if item not in common_elements]

        d = {"SMILES":gen_df_no_dupes_list}
        
        gen_df_no_dupes = pd.DataFrame(d)
        # double check no dupes 
        gen_df_no_dupes = gen_df_no_dupes.drop_duplicates()

        gen_df_no_dupes.to_excel(file_path)

        return gen_df_no_dupes
    
    else:
        gen_df.to_excel(file_path)
        return(gen_df)
    
def add_to_mol_db(
        gen_df,
        mol_db: str,
):
    """
    add generated molecules to master database
    """
    target_dir = Path("data")
    target_dir.mkdir(exist_ok=True)
    
    # load master db and get all smiles
    file_path = target_dir / mol_db
    master_df = pd.read_excel(file_path)
    all_smiles_list = master_df["SMILES"].to_list()
   
    canon_all_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in all_smiles_list]

    gen_df_list = gen_df["SMILES"].to_list()
    canon_gen_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in gen_df_list]

    set1, set2 = set(canon_gen_smiles), set(canon_all_smiles)

    # smiles in both sets
    common_elements = set1.intersection(set2)

    new_smiles = [item for item in gen_df_list if item not in common_elements]

    all_smiles = canon_all_smiles + new_smiles

    d = {"SMILES":all_smiles}

    all_smiles_df = pd.DataFrame(d)
    all_smiles_df.drop_duplicates()

    all_smiles_df.to_excel(file_path)

    return all_smiles_df

def plot_results(results: dict):
    """
    plot the results dictionary output from training loop
    """
    plt.figure(figsize=(7, 5))

    for key, values in results.items():
        plt.plot(values, label=key)

    # Labeling the plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()  # Display legend to show the label for each line

    plt.tight_layout()
    plt.show()

def create_writer(experiment_name: str, 
                  model_name: str,
                  fold: str,
                  extra: str=None):
    """
    create a summary writer for experiment tracking
    """
    from datetime import datetime
    import os
    
    time_stamp = datetime.now().strftime("%d-%m-%Y")

    if extra:
        log_dir = os.path.join("runs", time_stamp, experiment_name, model_name, fold, extra)

    else:
        log_dir = os.path.join("runs", time_stamp, experiment_name, model_name, fold)

    return SummaryWriter(log_dir=log_dir)

def save_pickle(results, pickle_file="data/default.pk1"):
    target_dir_path = Path(pickle_file)
    target_dir_path.parent.mkdir(parents=True, exist_ok=True)

    with open(pickle_file, "wb") as f:
        pickle.dump(results, f)

def load_pickle(pickle_file):
    with open(pickle_file, "rb") as f:
        loaded_results = pickle.load(f)
    return loaded_results

def remove_dummy_atoms(df):
    """
    generating molecules sometimes returns unphysical suggestions of dummy atoms
    so we remove this 
    """
    def contains_dummy(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False
        return any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())

    return df[~df['SMILES'].apply(contains_dummy)]

def filter_by_smarts(df, smarts):
    """
    remove molecules with designated substructures 
    """
    def has_substructure(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False
        pattern = Chem.MolFromSmarts(smarts)
        return mol.HasSubstructMatch(pattern)

    # Remove rows matching the SMARTS pattern
    return df[~df['SMILES'].apply(has_substructure)]

def display_molecule_grid(mols, df_filtered, max_mols=50, mols_per_row=5):
    """
    display a grid of molecule images with predicted transition temperatures.
    
    """
    grid_image = Draw.MolsToGridImage(
        mols[:max_mols],  # Limit to max_mols
        legends=[f"NF transition temp: {int(pred)}°C" for pred in df_filtered['PRED NF (C)'][:max_mols]],
        molsPerRow=mols_per_row,
        subImgSize=(300, 300),
        returnPNG=False
    )
    return grid_image

def create_mol_pdf(file_name, df, mols_per_page=12, mols_per_row=4):
    """
    Create PDF with molecular structures and predictions.
    Fixed to handle invalid SMILES and maintain proper alignment.
    """
    valid_data = []
    
    # Process molecules and keep only valid ones with aligned data
    for idx, smiles_str in enumerate(df['SMILES'].to_list()):
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is not None:
                # Compute 2D coordinates
                rdDepictor.Compute2DCoords(mol)
                
                # Store valid molecule with its corresponding data
                valid_data.append({
                    'mol': mol,
                    'smiles': smiles_str,
                    'pred': int(df['PRED NF (C)'].iloc[idx])
                })
            else:
                print(f"Warning: Invalid SMILES skipped: {smiles_str}")
                
        except Exception as e:
            print(f"Error processing SMILES '{smiles_str}': {str(e)}")
            continue
    
    if not valid_data:
        raise ValueError("No valid molecules found in the dataframe")
    
    print(f"Processing {len(valid_data)} valid molecules out of {len(df)} total")
    
    # Extract aligned lists from valid data
    mols = [item['mol'] for item in valid_data]
    preds = [item['pred'] for item in valid_data]
    smiles = [item['smiles'] for item in valid_data]
    
    # Create chunks (now all lists are guaranteed to be the same length)
    mol_chunks = [mols[x:x+mols_per_page] for x in range(0, len(mols), mols_per_page)]
    pred_chunks = [preds[x:x+mols_per_page] for x in range(0, len(preds), mols_per_page)]
    smiles_chunks = [smiles[x:x+mols_per_page] for x in range(0, len(smiles), mols_per_page)]
    
    grids = []
    
    try:
        for i in range(len(mol_chunks)):
            mol_chunk = mol_chunks[i]
            pred_chunk = pred_chunks[i]
            smiles_chunk = smiles_chunks[i]
            
            # Create legends
            legends_list = []
            for pred, smiles_str in zip(pred_chunk, smiles_chunk):
                legends_list.append(f"SMILES: {smiles_str}\nNF transition temp: {pred}°C")
            
            # Create grid image - Fixed parameter order
            grid_image = Draw.MolsToGridImage(
                mol_chunk,
                molsPerRow=mols_per_row,
                subImgSize=(500, 400),
                legends=legends_list,  # Move legends after other parameters
                returnPNG=False
            )
            grids.append(grid_image)
            
    except Exception as e:
        print(f"Error creating molecular grid: {str(e)}")
        raise
    
    # Save PDF
    try:
        # Ensure data directory exists
        os.makedirs("data/pdfs", exist_ok=True)
        pdf_path = f"data/pdfs/{file_name}.pdf"
        
        if len(grids) > 0:
            grids[0].save(
                pdf_path, 
                "PDF", 
                resolution=100.0, 
                save_all=True, 
                append_images=grids[1:] if len(grids) > 1 else []
            )
            print(f"PDF saved successfully: {pdf_path}")
        else:
            raise ValueError("No grids were created")
            
    except Exception as e:
        print(f"Error saving PDF: {str(e)}")
        raise
    
    return grids


def load_classifiers(model_filenames, target_dir="models/models_to_run", x_dim=79, device = 'cpu'):
    """
    Load models from filenames, parsing parameters from the filename patterns.
    
    Args:
        model_filenames: List of model filenames
        target_dir: Directory containing model files
        x_dim: Input dimension for models
        
    Returns:
        List of loaded models
    """
    n_pattern = re.compile(r'n_(\d+)')
    d_pattern = re.compile(r'd(0(?:\.\d+)?)')  # Updated to match "d0" or "d0.x"
    hdim_pattern = re.compile(r'hdim_(\d+)')
    
    
    loaded_models = []
    successful_loads = 0
    failed_loads = 0
    
    for filename in model_filenames:
        try:
            # Remove .pth extension to get model name
            model_name = filename.replace('.pth', '')
            
            # Extract model type (Classifier or EnhancedClassifier)
            if model_name.startswith('test_'):
                model_type = model_name.split('_')[1]
                prefix = 'test_'
            else:
                model_type = model_name.split('_')[0]
                prefix = ''
            
            # Extract parameters using regex
            n_match = n_pattern.search(filename)
            d_match = d_pattern.search(filename)
            hdim_match = hdim_pattern.search(filename)
            
            if n_match and d_match and hdim_match:
                n_conv_blocks = int(n_match.group(1))
                dropout = float(d_match.group(1))
                h_dim = int(hdim_match.group(1))
                
                # Instantiate the correct model type
                if "Classifier" in model_type and not any(x in model_type for x in ["Enhanced", "Transformer", "GCNClassifier", "GINClassifier", "GatedGraph"]):
                    model = models.Classifier(
                        x_dim=x_dim, 
                        h_dim=h_dim, 
                        n_conv_blocks=n_conv_blocks, 
                        dropout=dropout
                    )
                elif "EnhancedClassifier" in model_type:
                    model = models.EnhancedClassifier(
                        x_dim=x_dim, 
                        h_dim=h_dim, 
                        n_conv_blocks=n_conv_blocks, 
                        dropout=dropout
                    )

                elif "TransformerClassifier" in model_type:
                    model = models.TransformerClassifier(
                        x_dim=x_dim, 
                        h_dim=h_dim, 
                        n_conv_blocks=n_conv_blocks, 
                        dropout=dropout
                    )

                elif "GCNClassifier" in model_type:
                    model = models.GCNClassifier(
                        x_dim=x_dim, 
                        h_dim=h_dim, 
                        n_conv_blocks=n_conv_blocks, 
                        dropout=dropout
                    )
                elif "Gated" in model_type:
                    model = models.GatedGraphClassifier(
                        x_dim=x_dim, 
                        h_dim=h_dim, 
                        n_conv_blocks=n_conv_blocks, 
                        dropout=dropout
                    )
                elif "GINClassifier" in model_type:
                    model = models.GINClassifier(
                        x_dim=x_dim, 
                        h_dim=h_dim, 
                        n_conv_blocks=n_conv_blocks, 
                        dropout=dropout
                    )
                else:
                    print(f"Unknown model type in {filename}")
                    failed_loads += 1
                    continue
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    model.to(device)
                
                # Load model parameters
                load_params(model=model, target_dir=target_dir, model_name=filename)
                
                # Add model and its metadata to the list
                loaded_models.append({
                    'model': model,
                    'name': model_name,
                    'type': model_type,
                    'params': {
                        'n_conv_blocks': n_conv_blocks,
                        'dropout': dropout,
                        'h_dim': h_dim,
                        'x_dim': x_dim
                    }
                })
                
                successful_loads += 1
                print(f"Loaded {model_type} model: {filename}")
                
            else:
                print(f"Could not parse parameters from {filename}")
                failed_loads += 1
                
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            failed_loads += 1
    
    print(f"Successfully loaded {successful_loads} models, failed to load {failed_loads} models")
    return loaded_models

def load_predictors(model_filenames, target_dir="models/nf_pred_models_to_run", x_dim=79, device = 'cpu'):
    """
    Load models from filenames, parsing parameters from the filename patterns.
    
    Args:
        model_filenames: List of model filenames
        target_dir: Directory containing model files
        x_dim: Input dimension for models
        
    Returns:
        List of loaded models
    """
    n_pattern = re.compile(r'n_(\d+)')
    d_pattern = re.compile(r'd(0(?:\.\d+)?)')  # Updated to match "d0" or "d0.x"
    hdim_pattern = re.compile(r'hdim_(\d+)')
    
    
    loaded_models = []
    successful_loads = 0
    failed_loads = 0
    
    for filename in model_filenames:
        try:
            # Remove .pth extension to get model name
            model_name = filename.replace('.pth', '')
            
            # Extract model type (Classifier or EnhancedClassifier)
            if model_name.startswith('MP_'):
                model_type = model_name.split('_')[1]
                prefix = 'MP_'
            else:
                model_type = model_name.split('_')[0]
                prefix = ''
            
            # Extract parameters using regex
            n_match = n_pattern.search(filename)
            d_match = d_pattern.search(filename)
            hdim_match = hdim_pattern.search(filename)
            
            if n_match and d_match and hdim_match:
                n_conv_blocks = int(n_match.group(1))
                dropout = float(d_match.group(1))
                h_dim = int(hdim_match.group(1))
                
                # Instantiate the correct model type
                if "Predictor" in model_type and not any(x in model_type for x in ["Enhanced", "Transformer", "GCNPredictor", "GINPredictor", "GatedGraph"]):
                    model = models.Predictor(
                        x_dim=x_dim, 
                        h_dim=h_dim, 
                        n_conv_blocks=n_conv_blocks, 
                        dropout=dropout
                    )
                elif "EnhancedPredictor" in model_type:
                    model = models.EnhancedPredictor(
                        x_dim=x_dim, 
                        h_dim=h_dim, 
                        n_conv_blocks=n_conv_blocks, 
                        dropout=dropout
                    )

                elif "TransformerPredictor" in model_type:
                    model = models.TransformerPredictor(
                        x_dim=x_dim, 
                        h_dim=h_dim, 
                        n_conv_blocks=n_conv_blocks, 
                        dropout=dropout
                    )

                elif "GCNPredictor" in model_type:
                    model = models.GCNPredictor(
                        x_dim=x_dim, 
                        h_dim=h_dim, 
                        n_conv_blocks=n_conv_blocks, 
                        dropout=dropout
                    )
                elif "Gated" in model_type:
                    model = models.GatedGraphPredictor(
                        x_dim=x_dim, 
                        h_dim=h_dim, 
                        n_conv_blocks=n_conv_blocks, 
                        dropout=dropout
                    )
                elif "GINPredictor" in model_type:
                    model = models.GINPredictor(
                        x_dim=x_dim, 
                        h_dim=h_dim, 
                        n_conv_blocks=n_conv_blocks, 
                        dropout=dropout
                    )
                else:
                    print(f"Unknown model type in {filename}")
                    failed_loads += 1
                    continue
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    model.to(device)
                
                # Load model parameters
                load_params(model=model, target_dir=target_dir, model_name=filename)
                
                # Add model and its metadata to the list
                loaded_models.append({
                    'model': model,
                    'name': model_name,
                    'type': model_type,
                    'params': {
                        'n_conv_blocks': n_conv_blocks,
                        'dropout': dropout,
                        'h_dim': h_dim,
                        'x_dim': x_dim
                    }
                })
                
                successful_loads += 1
                print(f"Loaded {model_type} model: {filename}")
                
            else:
                print(f"Could not parse parameters from {filename}")
                failed_loads += 1
                
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            failed_loads += 1
    
    print(f"Successfully loaded {successful_loads} models, failed to load {failed_loads} models")
    return loaded_models

def match_models_with_predictions(model_filenames, pred_data_dir="predictions"):
    """
    Match model files with their corresponding prediction data files.
    
    Args:
        model_filenames: List of model filenames (.pth files)
        pred_data_dir: Directory containing prediction data files (.pkl files)
        
    Returns:
        Dictionary mapping model names to their prediction data files and DataFrames
    """
    # Define regex patterns for extracting model parameters - same as in load_models
    n_pattern = re.compile(r'n_(\d+)')
    d_pattern = re.compile(r'd(0(?:\.\d+)?)')
    hdim_pattern = re.compile(r'hdim_(\d+)')
    
    # Get all prediction data files
    pred_files = glob.glob(os.path.join(pred_data_dir, "pred_data*.pkl"))
    
    matches = {}
    
    for model_filename in model_filenames:
        # Remove .pth extension to get model name
        model_name = model_filename.replace('.pth', '')
        
        # Extract model parameters from filename
        n_match = n_pattern.search(model_filename)
        d_match = d_pattern.search(model_filename)
        hdim_match = hdim_pattern.search(model_filename)
        
        if n_match and d_match and hdim_match:
            n_conv_blocks = n_match.group(1)
            dropout = d_match.group(1)
            h_dim = hdim_match.group(1)
            
            # Determine model type
            if model_name.startswith('test_'):
                model_type = model_name.split('_')[1]
            else:
                model_type = model_name.split('_')[0]
            
            # Find matching prediction file based on model parameters
            matching_pred_files = []
            for pred_file in pred_files:
                # Look for prediction files that contain the same parameter values
                if (n_conv_blocks in pred_file and 
                    f"d{dropout}" in pred_file and 
                    f"hdim_{h_dim}" in pred_file and
                    model_type in pred_file):
                    matching_pred_files.append(pred_file)
            
            if matching_pred_files:
                # Load the first matching prediction file
                try:
                    pred_data = pd.read_pickle(matching_pred_files[0])
                    matches[model_name] = {
                        'prediction_file': matching_pred_files[0],
                        'prediction_data': pred_data
                    }
                    
                    # If multiple matches found, note them
                    if len(matching_pred_files) > 1:
                        matches[model_name]['multiple_matches'] = matching_pred_files
                except Exception as e:
                    print(f"Error loading {matching_pred_files[0]}: {str(e)}")
            else:
                matches[model_name] = {
                    'prediction_file': None,
                    'prediction_data': None,
                    'error': 'No matching prediction file found'
                }
    
    return matches

def display_model_prediction_matches(model_filenames, pred_data_dir="predictions"):
    """
    Display the matches between models and their prediction data files.
    
    Args:
        model_filenames: List of model filenames (.pth files)
        pred_data_dir: Directory containing prediction data files (.pkl files)
    """
    matches = match_models_with_predictions(model_filenames, pred_data_dir)
    
    print(f"{'Model Name':<50} | {'Prediction File':<50}")
    print("-" * 102)
    
    for model_name, match_info in matches.items():
        pred_file = match_info['prediction_file']
        if pred_file:
            pred_filename = os.path.basename(pred_file)
            print(f"{model_name:<50} | {pred_filename:<50}")
        else:
            print(f"{model_name:<50} | {'No matching prediction file found':<50}")