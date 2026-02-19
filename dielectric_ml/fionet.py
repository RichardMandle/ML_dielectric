# fionet - file i/o things for neural network

import pandas as pd
import numpy as np
import re

def load_data(filename, drop_inplace=True, reset_index=True, **kwargs):
    """
    lets have a little loading function with **kwargs for extra power.
    
    args:
    - filename (str): path to the file; can handle xlsx and CSV (csv not tested yet)
    - drop_inplace (bool): Drop empty rows (True/False); you want to do this.
    - reset_index (bool): reset the row index after getting rid of empties (True/False); I think you want to do this.
    - **kwargs: pass any arguments you like to pandas read functions (e.g., sheet_name for excel).
    
    returns:
    - df (pd.df): returns the pandas df object back
    """
    try:
        # Determine file type and load data
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(filename, **kwargs)
        elif filename.endswith('.csv'):
            df = pd.read_csv(filename, **kwargs)
        else:
            raise ValueError("This file type is unsuported. Please provide an .xlsx or .csvfile.")
        
        if drop_inplace:
            df.dropna(how='all', inplace=True)

        if reset_index:
            df.reset_index(drop=True, inplace=True)
        
        return df
    
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def remove_zero_columns(data):
    non_zero_columns = (data != 0).any(axis=0)
    return data[:, non_zero_columns]


def prepare_data(df, fingerprints, transition = 'i', remove_zero_cols = True, remove_zero_rows = True, gen_bool_label = False, use_k = True, verbose = True):
    '''
    wrapper function that takes a data frame (df) and gets the transiiton temperatures of the transition
    of interest (transition = i). Finds zero entries (remove_zero_rows = True), and uses a mask to remove
    entries without transition data (transition temperateures and fingerprints)
    
    args:
        df - dataframe that contains transition data in column
        fingerprints - fingerprint data
        traisition (=i) - transition of interest; case insensitive
        remove_zero_cols - remove columns of zeros from fingerprints (dud data)
        remove_zero_rows - filter out compounds (fingerprints + transition) where transition above is absent
        gen_bool_label - returns a boolean array indicating if a given material has the phase (transition) in question (==1) or not (==0)
        use_k - use kelvin (converts to K from deg C)
        verbose (=True) - print extra output
    returns:
        labels - (optional) a boolean array indicating if a transition occurs or not 
        trans_temps - the (optionally filtered) temperature of the transition in question
        fingerprints - the (optionally filtered) fingerprint of the molecule in question.
        
    example usage:
        
        flabels, temps, frags = fionet.prepare_data(df, 
                                     fingerprints_morgan, 
                                     transition = 'Nf',
                                    gen_bool_label = True)
    '''
        
    transition_temperatures = parse_transitions(df['Transitions / dC'], use_k = use_k)
    
    trans_temps = get_transition_temps(transition_temperatures, transition.lower()) # lowercase!
    
    if verbose:
        print(f'Fingeprint shape: {np.shape(fingerprints)}')
        print(f'Num. of transitions: {np.shape(trans_temps)[0]}')
    
    if remove_zero_cols:
        fingerprints = remove_zero_columns(fingerprints)
        
        if verbose:
            print(f'\nFingerprint shape after removing zero columns: {np.shape(fingerprints)}\n')
    
    if gen_bool_label:
        labels = [] 
        for temp in trans_temps:
            if temp != 0:  # if the transition is there
                labels.append(1)    # n transition == 0
            else:
                labels.append(0)    # Transition occurs; ==1
        
        labels = np.array(labels)
        
        if verbose:
            print(f'Total of {np.sum(labels == 1)} entries with "{transition}" transition in data')
            print(f'Total of {np.sum(labels == 0)} entries without "{transition}" transition in data')
            
        return labels, np.array(trans_temps), np.array(fingerprints)
    
    if remove_zero_rows:
        valid_indices = trans_temps.flatten() != 0
        fingerprints = fingerprints[valid_indices]
        trans_temps = trans_temps[valid_indices]
        
        if verbose:
            print(f'Fingerprints shape after removing zeros: {np.shape(fingerprints)}')
            print(f'Num. of transitions after removing zeros: {np.shape(trans_temps)[0]}')

    return np.array(trans_temps), np.array(fingerprints)

def get_transition_temps(transition_temperatures, transitions=['nf'], print_output=True):
    '''
    Extract the highest transition temperature for a list of specified transitions,
    even when the transitions are part of more complex key strings (e.g., "K - Nf").

    Args:
        transition_temperatures: List of dictionaries containing transitions.
        transitions: List of transition strings to look for.
        print_output: If True, prints a summary report.

    Returns:
        temps: Array of the highest temperatures corresponding to the specified transitions.
    '''
    temps = []  # Our temperatures
    found_transitions = set()  # Track found transitions

    transitions = [t.lower() for t in transitions]

    for i, trans_dict in enumerate(transition_temperatures):
        max_temp = float('-inf')  # start with the lowest possible value to ensure capturing of all temps
        for key in trans_dict:
            key_lower = key.lower()  # keep the key to lowercase for comparison
            for transition in transitions:
                if transition in key_lower:  # do a general check to capture any position of the transition in the key
                    if not pd.isna(trans_dict[key]):  # ensure the temperature is not NaN
                        #print(f"Matched transition no {i}: {key} with temperature: {trans_dict[key]}. Original {transition}") # a useful debug line
                        max_temp = max(max_temp, trans_dict[key])  # note the highest temperature
                        found_transitions.add(key)  # keep the original case for the report

        temps.append(max_temp if max_temp != float('-inf') else 0)  # Convert '-inf' back to 0 if no valid temp found

    temps = np.array(temps).reshape(-1, 1)
    
    if print_output:       
        print(f"{len(transition_temperatures)} were parsed, with {np.sum(temps !=0)} hits for {transitions} (only the highest transition temperature is recorded!)")

    return temps
    	
def parse_transitions(transitions, use_k=True):
    '''
    Takes transition data like K 123 N 456 I and gets the individual transition temperatures
    
    we don't use it here, its a legacy from the NF stuff, but it might be useful one day.
    '''
    parsed_data = []
    for trans in transitions:
        if pd.notna(trans):
            trans = str(trans)
            phases = re.findall(r'[A-Za-z]+', trans)
            temps = re.findall(r'-?[\d.]+', trans)  # mod. to handle negative temperatures
            
            temps_k = [np.round(float(temp) + (273.15 * use_k), 1) for temp in temps]
            
            trans_dict = {}
            
            if phases[0] == 'K':
                if len(temps_k) == 0 or not re.match(r'-?[\d.]+', trans.split('K', 1)[1].strip()):
                    trans_dict['K'] = np.nan
                    phases = phases[1:]  # lose 'K' since it's handled separately
                else:
                    trans_dict['K'] = temps_k[0]
                    temps_k = temps_k[1:]  # remove the first temperature associated with 'K', i.e. meting point
                    phases = phases[1:]  # remove 'K' as it's now processed (above!)

            for i in range(len(temps_k)):
                if i < len(phases) - 1:
                    transition_key = f"{phases[i]} - {phases[i+1]}"
                    trans_dict[transition_key] = temps_k[i]
            
            parsed_data.append(trans_dict)
        else:
            parsed_data.append({})
            
    return parsed_data
