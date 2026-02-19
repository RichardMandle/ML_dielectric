# prednet - functions for predicting on molecules generated/reported elsewhere

import numpy as np
import pandas as pd
import torch
from PIL import Image

import skfp.fingerprints as fp # presumably we'll need this for fingerprinting.

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools

# turn on rendering of RDKit molecules in DataFrame
# does this want to be here or the notebook?
PandasTools.ChangeMoleculeRendering(renderer='PNG')

# import our own modules
import dielectric_ml


###
#WARNING
###

# A lot of this code is pretty old; I doubt it works with the newer versions of chemnet/neurnet...
'''
list of functions:
    
    smi2df - converts a list of smiles strings to a pd.df, generates conformers and fingerprints
    make_prediction - takes a list of smiles, fingerprints etc. and makes a prediction
    predict_from_smiles - wrapper function; takes a list of smiles and a model and interacts with 2 funcs above.
'''

def smi2df(smiles, method='ETKDGv3', num_conformers=3, max_steps=1000, threshold=1e-4):
    '''
    take a list of smiles strings, generates a dataframe, generates conformer information.

    args:
        smiles (list)   -   list of SMILES strings.
        method (str)    -   conformer generation method.
        num_conformers (int) - no. of conformers to generate while looking for minimum
        max_steps (int) -   maximum optimization steps for conformer generation.
        threshold (float)-  energy convergence threshold for conformer generation.

    returns:
        df  -   a pd.dataframe with SMILES strings and generated conformers.
    '''
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        
    df = pd.DataFrame({'SMILES': smiles, 'Molecule': mols})

    if not smiles:    # check if SMILES are valid before generating conformers
        raise ValueError("No SMILES provided to generate dataframe.")
    
    dielectric_ml.chemnet.generate_low_energy_conformers(df, smiles_column='SMILES',
                                           method=method,
                                           num_conformers=num_conformers,
                                           max_steps=max_steps,
                                           threshold=threshold)
    
    PandasTools.AddMoleculeColumnToFrame(df, 'SMILES', 'Molecule', includeFingerprints=False)

    return df
    
def make_prediction(smiles, model, fingerprint, scaler_x=None, scaler_y=None, fp_type=None, optimism=None):
    '''
    args:
        smiles: List of SMILES strings.
        model: Trained PyTorch model for prediction.
        fingerprint: A dict or np.array of fingerprints for prediction.
        scaler_x: Optional scaler used to normalize the input fingerprints.
        scaler_y: Optional scaler used to scale the output back to real-world values (e.g., temperature).
        fp_type: Optional, the fingerprint type if a dict of fingerprints is passed.
        optimism: Float (0 -> 1); larger values return higher transition temperatures.
        draw_output: Boolean; whether to draw molecular images with predictions as labels.

    returns:
        prediction: The predicted transition temperatures.
        img (optional): An image of molecules with predicted values in the labels.
    '''

    fprint_tensor = np.array(fingerprint)
    print(np.shape(fprint_tensor))
    # inject the optimism bias
    if optimism:
        print("biasing")
        bias_feature = np.ones((fprint_tensor.shape[0], 1)) * optimism
        fprint_tensor = np.concatenate((fprint_tensor, bias_feature), axis=1)
    print(np.shape(fprint_tensor))
    if scaler_x is not None: # apply the x-scaler; note, it'll be terrible if not provided.
        fprint_tensor = scaler_x.transform(fprint_tensor)

    fprint_tensor = torch.from_numpy(fprint_tensor).float()

    model.eval()
    with torch.no_grad():
        output_tensor = model(fprint_tensor)
        output = output_tensor.detach().numpy()

    if scaler_y is not None:
        prediction = scaler_y.inverse_transform(output).flatten()
    else:
        prediction = output.flatten()

    return prediction

    
def predict_from_smiles(smiles, model, fp_type, scaler_x, scaler_y, optimism=0, sort_by_pred = True, **kwargs):
    '''
    wrapper function to predict transition temperatures for a list of SMILES strings.
    
    args:
        smiles      - list of SMILES strings to make predictions on.
        model       - pre-trained model for prediction.
        fp_type     - the type of fingerprint to use for prediction (e.g., 'mordred_fingerprint_3D').
        scaler_y    - scaler used for converting outputs back to real values. # NOT CURRENTLY USED
        optimism    - float value (0 --> 1) to adjust the optimism in prediction.
        sort_by_pred- Bool; do you want to sort the returned dataframe so the highest prediction is first?
        **kwargs    - additional keyword arguments for fingerprint generation, etc.

    returns:
        pred_df  - the predictions, smiles, mol objects, conformer etc. in a handy pd.df
        
    example:
        smi3 = ['CCCCC','c0ccccc0','CCCCCc0ccc(c1ccc(C#N)cc1)cc0','c1c(c(ccc1OC)C(=O)Oc1ccc(cc1)C(=O)Oc1ccc(cc1)[N+](=O)[O-])OC'] # some random smiles entries
        fp_type = 'mordred_fingerprint_3D' # must be THE SAME as the one the network is trained on
        optimism = 1 # do you feel lucky?
        pred_df = prednet.predict_from_smiles(smi3, model=best_model, fp_type=fp_type, scaler_x = best_model_scaler_x, scaler_y=best_model_scaler_y, optimism=optimism) # call the prediction engine
        pred_df # will display the pred_df in the notebook window.
        
    2nd example:
        as above, but:
        PandasTools.FrameToGridImage(pred_df, column='Molecule', legendsCol="prediction ° C", molsPerRow=6) # displays a grid image from the df, with predictions as labels
    '''
    
    pred_df = smi2df(smiles, **kwargs)

    pred_fps = dielectric_ml.chemnet.get_fingerprints(pred_df, smiles_column='SMILES', 
                                        conformer_column='min_e_conf', 
                                        n_jobs=kwargs.get('n_jobs', 1), 
                                        fp_size=kwargs.get('fp_size', 512), 
                                        fp_type=fp_type, 
                                        use_3D=kwargs.get('use_3D', True))

    # use chemnet.clean_fprints_targets with a np.zeros array which just mimics the expected array of temperatures; which we discard the output of this with "_"
    filtered_pred_fps, _ = dielectric_ml.chemnet.clean_fprints_targets(pred_fps[fp_type], 
                                        np.zeros(np.shape(pred_fps[fp_type])[1]), print_output=True)
    
    print(f" Length of fingerprint: {np.shape(filtered_pred_fps)}")
    print(model[0].in_features)
    
    pred = make_prediction(pred_df['SMILES'], model, filtered_pred_fps, scaler_x,
                                        scaler_y, fp_type=fp_type, optimism=optimism)
    
    
    pred_df.loc[:, 'prediction'] = np.clip(pred, a_min = -273, a_max = 1000)
    
    if np.sum(pred_df['conformer_error']) == 0:
        pred_df = pred_df[['prediction', 'SMILES', 'Molecule', 'min_e_conf']] # reorder for nice visual, bin the conformer_error if all empty
    elif np.sum(pred_df['conformer_error']) != 0:
        print(f'***Warning***\nConformer error found; see returned DataFrame')
        pred_df = pred_df[['prediction', 'SMILES', 'Molecule', 'min_e_conf', 'conformer_error']] # as above but warn about conformer errors
        
    if sort_by_pred:
        pred_df = pred_df.sort_values(by='prediction', ascending=False) # sort biggest prediction to smallest
    
    PandasTools.AddMoleculeColumnToFrame(pred_df, 'SMILES', 'Molecule', includeFingerprints=False)
    return pred_df
