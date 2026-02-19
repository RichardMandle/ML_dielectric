# neurnet - functions for network

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import copy

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold # new addition
from sklearn.model_selection import train_test_split

# import our own modules
import dielectric_ml

'''
List of functions

get_tensors                 - convert arrays to torch.tensor
predict                     - make a prediction for a given set of data, using the model, and using the y-scaler to get back to "real" numbers
initialize_weights          - initialize the weights of the model
train_model_kfold           - k-fold model training
create_simple_model         - creates a network/model
analyse_results             - analyse and plot data across many types of fingerprint
get_basic_layers_config     - make a standard configuration of layers, used if layers_config isn't defined
train_fingerprint_model     - a wrapper function to train a model using fingerprint data and transition temperatures; accepts kwargs for flexibility.

'''

def get_tensors(X,y, remove_zero_cols = False, print_output = True):
    '''
    Prepare data as tensors; remove zero entries if remove_zero_cols = True
    returns tensors and scaler for Y (which is normalised, so we need this to convert back to real values)
    '''
    if remove_zero_cols:
        print("Removing zero columns...")
        X = dielectric_ml.fionet.remove_zero_columns(X)

    scaler_x = StandardScaler()
    X_normalised = scaler_x.fit_transform(X)
    scaler_y = StandardScaler()
    y_normalised = scaler_y.fit_transform(y.reshape(-1, 1))

    X_tensor = torch.tensor(X_normalised, dtype=torch.float32)
    y_tensor = torch.tensor(y_normalised, dtype=torch.float32)

    if print_output:
        print(f'Initial X shape: {np.shape(X)}: Final (normalised) X shape: {np.shape(X_normalised)}')
        print(f'Initial y shape: {np.shape(y)}: Final (normalised) y shape: {np.shape(y_normalised)}')

    return X_tensor, y_tensor, scaler_x, scaler_y

def tensors_from_scalers(X, y, scaler_x, scaler_y):
    '''
    Convert arrays to torch tensors using provided scalers (no fitting).
    '''
    Xn = scaler_x.transform(X)
    yn = scaler_y.transform(y.reshape(-1, 1))
    X_tensor = torch.tensor(Xn, dtype=torch.float32)
    y_tensor = torch.tensor(yn, dtype=torch.float32)
    return X_tensor, y_tensor

def predict(model, data_loader):
    '''
    predict values using a model (model) for data (data_loader)
    '''
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            predictions.append(outputs.detach().cpu().numpy())
            
            actuals.append(targets.detach().cpu().numpy())

    return np.concatenate(predictions), np.concatenate(actuals)

def initialize_weights(m, init_type='xavier_uniform'):
    '''
    slightly more complex tool for initialization of weights using different methods.

    args:
        m = model
        init_type = type of initialization; https://pytorch.org/docs/stable/nn.init.html
    '''
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        if init_type == 'xavier_uniform': #https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
            nn.init.xavier_uniform_(m.weight)
        elif init_type == 'xavier_normal': #https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
            nn.init.xavier_normal_(m.weight)
        elif init_type == 'kaiming_uniform': #https://arxiv.org/abs/1502.01852
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif init_type == 'kaiming_normal': #https://arxiv.org/abs/1502.01852
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif init_type == 'zeros': # just zeros
            nn.init.zeros_(m.weight)
        elif init_type == 'custom': # experimental stuff
            nn.init.normal_(m.weight, mean=0, std=0.01)

        # bias, if present
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def get_rmse(model, data_loader, scaler):
    '''
    computes the RMSE for a model with the provided data_loader and scaler.

    args:
        model       -   the model we'll evlauate
        data_loader -   the DataLoader object containing the dataset (could be training, validation, hold out etc.)
        scaler      -   a standard scaler used to transform the target data (needed to unscale the predictions).

    returns:
        rmse        - root mean square error.
        p_unscaled  - unscaled predictions from the model.
        a_unscaled  - unscaled ground truth values.
    '''
    pred, acts = predict(model, data_loader)
    p_unscaled = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    a_unscaled = scaler.inverse_transform(acts.reshape(-1, 1)).flatten()
    rmse = np.sqrt(np.mean((p_unscaled - a_unscaled) ** 2))

    return rmse, p_unscaled, a_unscaled

def train_model_kfold(fingerprints, nf_temps, model, criterion=None, optimiser_class=None, fold_type='kf', init_type='xavier_uniform', test_split_size=0.2, num_epochs=20, k=5, n_repeats = 3, batch_size=32, learn_rate=0.002, weight_decay=1e-5, style='scatter', print_output=True, plot_output=True, print_output_frequency=0.2, limits=[300, 500], patience=50):
    '''
    training function: performs data splitting, k-fold cross-validation, and final testing.
    returns the best model and various metrics.


    args:
        fingerprints: Input features (e.g., molecular fingerprints).
        nf_temps: Target transition temperatures.
        model: Neural network model to train.
        criterion: Loss function; (defaults to None, loads nn.MSELoss as an object)
        optimiser_class: Optimiser class (defaults to None, loads optim.Adam in that case).
        fold_type: either kf (k-fold) or rkf (repeated k-fold)
        init_type: type of initialization to use, defaults to 'xavier_uniform', see https://pytorch.org/docs/stable/nn.init.html
        num_epochs: Number of epochs to train.
        k: Number of folds for cross-validation.
        n_repeats: when using rkf (repeated k-fold) use this many repeats
        batch_size: Batch size for DataLoader.
        learn_rate: Learning rate for the optimiser.
        weight_decay: Weight decay (L2 regularization) for the optimiser.
        style: Plot style (e.g., 'scatter').
        print_output: Whether to print output during training (and also when getting tensors). if 'verbose', prints even more
        print_output_frequency: the frequency with which to report on the training in an epoch (default = 0.2)
        plot_output: Whether to generate plots during training.
        limits: the limits used on our x=y part of the plot.
    '''

    if criterion is None:
        criterion = nn.MSELoss()

    if optimiser_class is None:
        optimiser_class = optim.Adam  # Default optimiser
        if print_output:
            print('Using default optim.Adam optimiser')

    X_train_val, X_test, y_train_val, y_test = train_test_split(fingerprints, nf_temps, test_size=test_split_size, random_state=32)

    X_train_val = np.asarray(X_train_val)
    y_train_val = np.asarray(y_train_val)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    if fold_type == 'kf':
        kf = KFold(n_splits=k, shuffle=True, random_state=1)
    elif fold_type == 'rkf':
        print(f'Repeated K-Fold splitting; will run {k * n_repeats} folds total')
        kf = RepeatedKFold(n_splits=k, n_repeats = n_repeats, random_state=1)
    else:
        print(f'You didn\'t provide a valid fold_type; you can have: "kf" | "rkf"; you provided: "{fold_type}"')
        print(f'Defaulting to KFold')
        kf = KFold(n_splits=k, shuffle=True, random_state=1)

    avg_val_rmse = 0
    best_model_global = None
    best_val_rmse_global = float('inf')  # set to a very large (inf) value globally, i.e. across folds
    best_fold_global = 0
    best_model_state_global = None
    best_model_scaler_y = None
    best_model_scaler_x = None

    all_train_rmses = []
    all_val_rmses = []
    all_repeats_train_rmses = []
    all_repeats_val_rmses = []

    fold = 1
    for train_index, val_index in kf.split(X_train_val):
        print(f'Fold {fold}')
        model.apply(lambda m: initialize_weights(m, init_type=init_type))  # reset weights

        all_train_rmses, all_val_rmses = [],[] # reset RMSE lists

        X_train_raw, X_val_raw = X_train_val[train_index], X_train_val[val_index]
        y_train_raw, y_val_raw = y_train_val[train_index], y_train_val[val_index]

        scaler_x = StandardScaler().fit(X_train_raw)
        scaler_y = StandardScaler().fit(y_train_raw.reshape(-1, 1))

        X_train, y_train = tensors_from_scalers(X_train_raw, y_train_raw, scaler_x, scaler_y)
        X_val, y_val = tensors_from_scalers(X_val_raw, y_val_raw, scaler_x, scaler_y)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        #to do true / false switch in train_lodaer = ...
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        optimiser = optimiser_class(model.parameters(), lr=learn_rate, weight_decay=weight_decay)

        train_rmses = []
        val_rmses = []

        best_val_rmse_fold = float('inf')
        best_model_state = copy.deepcopy(model.state_dict())
        bad_epochs = 0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                optimiser.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()
                running_loss += loss.item() * inputs.size(0)

            train_rmse, train_preds_unscaled, train_actuals_unscaled = get_rmse(model, train_loader, scaler_y) # get RMSE via get_rmse
            all_train_rmses.append(train_rmse)

            model.eval()
            with torch.no_grad():
                val_rmse, val_preds_unscaled, val_actuals_unscaled = get_rmse(model, val_loader, scaler_y) # again, get RMSE via get_rmse
            all_val_rmses.append(val_rmse)

            if val_rmse < best_val_rmse_fold:
                best_val_rmse_fold = val_rmse
                best_model_state = copy.deepcopy(model.state_dict())   # save best model state dict (we'll need this to reload the best in epoch)
                bad_epochs = 0
            else:
                bad_epochs += 1

            if patience is not None and bad_epochs >= patience:
                if print_output:
                    print(f'Early stopping at epoch {epoch+1}/{num_epochs} (patience={patience})')
                break

            if print_output and (epoch + 1) % int(num_epochs * print_output_frequency) == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}')

        model.load_state_dict(best_model_state) # retrieve the best model in the epoch run from the state dict

        train_rmse, train_preds_unscaled, train_actuals_unscaled = get_rmse(model, train_loader, scaler_y)
        val_rmse, val_preds_unscaled, val_actuals_unscaled = get_rmse(model, val_loader, scaler_y)

        if plot_output:
            dielectric_ml.plotnet.plot_model(train_actuals_unscaled, train_preds_unscaled, val_actuals_unscaled, val_preds_unscaled, train_rmse, best_val_rmse_fold, fold, all_train_rmses, all_val_rmses, gridsize=25, cmap='Blues', plot_xy=True, style=style, limits=limits)

        # update the best model and RMSE globally if they are better than the global best
        if best_val_rmse_fold < best_val_rmse_global:
            best_val_rmse_global = best_val_rmse_fold
            best_fold_global = fold
            best_model_state_global = copy.deepcopy(best_model_state)
            best_model_scaler_y = scaler_y
            best_model_scaler_x = scaler_x

        all_repeats_train_rmses.append(train_rmse)
        all_repeats_val_rmses.append(val_rmse)
        fold += 1

    if fold_type == 'rkf':# calculate and report RMSE across all folds and repeats
        print(all_repeats_train_rmses)
        avg_train_rmse = np.mean(all_repeats_train_rmses)
        avg_val_rmse = np.mean(all_repeats_val_rmses)

        print(f'Average train RMSE across repeats and folds: {avg_train_rmse:.4f}')
        print(f'Average validation RMSE across repeats and folds: {avg_val_rmse:.4f}')
    else:
        avg_val_rmse = np.mean(all_repeats_val_rmses) if len(all_repeats_val_rmses) > 0 else np.nan

    fold_train_rmses = np.asarray(all_repeats_train_rmses, dtype=float)
    fold_val_rmses   = np.asarray(all_repeats_val_rmses, dtype=float)

    # ddof=1 gives sample std dev; if only 1 fold, then std=0.
    std_train_rmse = float(np.nanstd(fold_train_rmses, ddof=1)) if fold_train_rmses.size > 1 else 0.0
    std_val_rmse   = float(np.nanstd(fold_val_rmses,   ddof=1)) if fold_val_rmses.size > 1 else 0.0

    if print_output:
        n_folds_total = len(all_repeats_val_rmses)
        print(f'Validation RMSE across folds: mean={avg_val_rmse:.4f} ± {std_val_rmse:.4f} (n={n_folds_total})')

    if best_model_state_global is not None:
        model.load_state_dict(best_model_state_global)
        best_model_global = model
    else:
        best_model_global = model

    # now we will evaluate the best global model on the holdout test set
    X_test_tensor, y_test_tensor = tensors_from_scalers(X_test, y_test, best_model_scaler_x, best_model_scaler_y)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    best_model_global.eval()
    with torch.no_grad():
        test_rmse, test_preds_unscaled, test_actuals_unscaled = get_rmse(best_model_global, test_loader, best_model_scaler_y)

    print(f'Final holdout test set RMSE with best model (from fold {best_fold_global}) has RMSE: {test_rmse:.4f}')
    if plot_output:
        dielectric_ml.plotnet.plot_test_results(test_actuals_unscaled, test_preds_unscaled, test_rmse, gridsize=25, cmap='Blues', plot_xy=True, style=style, limits=limits)

    return (best_model_global,
            avg_val_rmse,
            test_rmse,
            all_train_rmses,
            all_val_rmses,
            best_model_scaler_x,
            best_model_scaler_y,
            std_val_rmse,          # NEW
            fold_val_rmses.tolist(), # NEW (raw per-fold values; handy for error bars / boxplots)
            test_actuals_unscaled,
            test_preds_unscaled,
           )
def create_simple_model(input_data, layer_config, output_size=1):
    '''
    Create a simple neural network model
    
    Basically this is just making life easier in the notebook rather than doing anything fancy.
    
    args:
    input_data - the input data we'll train on, just used for size of first bit of network
    layers_config - a list of layers and relevant parameters, see example
    output_size - the output size, defaults = 1

    returns:
    nn.Sequential(*layers) - a sequential network model

    example of layer_config:

        layers_config = [
        (nn.Linear, {'out_size': 64}),
        (nn.ReLU, {}),
        (nn.Dropout, {'p': 0.3}),
        (nn.Linear, {'out_size': 32}),
        (nn.BatchNorm1d, {}),
        (nn.ReLU, {}),]

    example of use:

        model = create_simple_model(input_data = fprints,
                                    layers_config = layers_config,
                                    output_size = 1)
    '''
    layers = []
    current_input_size = np.shape(input_data)[1]

    for layer_entry in layer_config:
        layer_class_or_instance, layer_params = layer_entry

        if isinstance(layer_class_or_instance, type):  # Check if it's a class
            if layer_class_or_instance == nn.Linear:
                out_size = layer_params.get('out_size', current_input_size // 2)
                layer_instance = layer_class_or_instance(current_input_size, out_size)
                current_input_size = out_size
            
            elif layer_class_or_instance == nn.BatchNorm1d:
                # for BatchNorm1d, we need to pass the current input size as num_features
                layer_instance = layer_class_or_instance(current_input_size)
            
            else:
                layer_instance = layer_class_or_instance(**layer_params)
        else:
            # It's already an instance, so just use it directly
            layer_instance = layer_class_or_instance

        layers.append(layer_instance)

    layers.append(nn.Linear(current_input_size, output_size))  # Final output layer
    return nn.Sequential(*layers)

def analyse_results(results_dict, threshold=None):
    """
    analyses the results dictionary from ??? to compute mean, std dev, and other statistics for
    each fingerprint type.

    args:
    - results_dict: dict of the results.
    - threshold: (optional!) a threshold to exclude outliers.

    returns:
    - analysis_dict: A dict with computed statistics.
    """
    analysis_dict = {}

    for fp_type, metrics in results_dict.items():
        if threshold:
            filtered_val_rmses = [val_rmse for val_rmse in metrics['all_val_rmses'] if val_rmse <= threshold]
            filtered_train_rmses = [train_rmse for train_rmse in metrics['all_train_rmses'] if train_rmse <= threshold]
        else:
            filtered_val_rmses = metrics['all_val_rmses']
            filtered_train_rmses = metrics['all_train_rmses']

        mean_val_rmse = np.mean(filtered_val_rmses) if filtered_val_rmses else np.nan
        std_val_rmse = np.std(filtered_val_rmses) if filtered_val_rmses else np.nan
        mean_train_rmse = np.mean(filtered_train_rmses) if filtered_train_rmses else np.nan
        std_train_rmse = np.std(filtered_train_rmses) if filtered_train_rmses else np.nan

        analysis_dict[fp_type] = {
            'mean_val_rmse': mean_val_rmse,
            'std_val_rmse': std_val_rmse,
            'test_rmse': metrics['test_rmse'],
            'mean_train_rmse': mean_train_rmse,
            'std_train_rmse': std_train_rmse,
        }

    return analysis_dict

def get_basic_layers_config(l1_size = 32, l2_size = 8, dropout = 0.25):
    '''
    Simple function for returning a basic layer configuration if the user doesn't provide one.
    '''

    layers_config = [
    (nn.Linear, {'out_size': l1_size}),
    (nn.ReLU, {}),
    (nn.Dropout, {'p': dropout}),
    (nn.Linear, {'out_size': l2_size}),
    (nn.BatchNorm1d, {}),  # no need to pass parameters; it uses current input size!
    (nn.ReLU, {}),
    ]

    return layers_config

def train_fingerprint_model(dataframe, fingerprints, transition_of_interest, layers_config = None, fold_type='kf', fp_type=None, **kwargs):
    """
    a wrapper function to train a model using fingerprint data and transition temperatures.

    args:
        dataframe (pd.df): DataFrame containing the transition data.
        fingerprints (dict): Dictionary of fingerprints.
        transition_of_interest (list): List of transitions to focus on.
        type: the type of split; either kfold (kf) or repeated kfold (rkf)
        fp_type (str, optional): Type of fingerprint to use from the fps dict. Defaults to None (selects the 24th key).
        **kwargs: Additional keyword arguments for the training function.

    Returns:
        best_model: Trained model with the best performance.
        avg_val_rmse: Average validation RMSE over k-fold cross-validation.
        test_rmse: RMSE on the holdout test set.
        all_train_rmses: List of RMSEs for each training fold.
        all_val_rmses: List of RMSEs for each validation fold.
        scaler_y: the scaler to get back to real numbers

    example use:

    normal stuff
    best_model, avg_val_rmse, test_rmse, all_train_rmses, all_val_rmses = train_fingerprint_model(dataframe = df,
                                                                                              fingerprints = fps,
                                                                                              transition_of_interest = ['nf'],
                                                                                              layers_config = layers_config,
                                                                                              fp_type = None,
                                                                                              num_epochs = 100)

    training over a lot of fingerprints in a loop with scoring:
    results_dict = {}
    layers_config = neurnet.get_basic_layers_config()

    for i in range(19,len(fps.keys())):
        fp_type = list(fps.keys())[i]
        print(fp_type)
        best_model, avg_val_rmse, test_rmse, all_train_rmses, all_val_rmses = neurnet.train_fingerprint_model(dataframe = df,
                                                                                                      fingerprints = fps,
                                                                                                      transition_of_interest = ['nf'],
                                                                                                      layers_config = layers_config,
                                                                                                      k = 4,
                                                                                                      fp_type = fp_type,
                                                                                                      num_epochs = 100,
                                                                                                      plot_output = False)
        results_dict[fp_type] = {
        'avg_val_rmse': avg_val_rmse,
        'test_rmse': test_rmse,
        'all_train_rmses': all_train_rmses,
        'all_val_rmses': all_val_rmses
        }

    # convert the results dictionary to a Pandas DataFrame for easier analysis
    results_df = pd.DataFrame(results_dict).transpose()

    # analyse and visualise
    analysis_dict = neurnet.analyse_results(results_dict, threshold=100)
    plotnet.plot_rmse_comparison(analysis_dict)
    """

    # load and parse transition data
    parsed_transitions = dielectric_ml.fionet.parse_transitions(dataframe['Transitions / dC'], use_k=True)
    temps = dielectric_ml.fionet.get_transition_temps(parsed_transitions, transitions=[t.lower() for t in transition_of_interest])

    if layers_config is None:
        layers_config = dielectric_ml.neurnet.get_basic_layers_config() # if no layer_config is passed, retrieve a default one from neurnet.get_basic_layers_config():

    if fp_type is None:
        fp_type = list(fingerprints.keys())[0] # select the fingerprint type to use, defaults to the 1st key if not provided
        print(f'Using default fp_type of {fp_type}')

    if fp_type not in fingerprints.keys(): # if the requested fingerprint isn't in fingerprints, then do print an error...
        print(f'\n*** WARNING ***\nfingerprint type {fp_type} is not in fingerprints.keys(); available types are:')
        for i, key in enumerate(fingerprints.keys()): # send a list of what they can use
            print(f'[{i}] {key}')
        print(f'defaulting to {list(fingerprints.keys())[0]}\n *** WARNING ENDS ***')
        fp_type = list(fingerprints.keys())[0] # and start using the first one, why not.

    # filter out cases where fingerprints or temps are None
    fprints, temps = dielectric_ml.chemnet.clean_fprints_targets(fingerprints[fp_type],
                                                   temps,
                                                   remove_zero_targets=kwargs.get('remove_zero_targets', False),
                                                   add_bool_fingerprint=kwargs.get('add_bool_fingerprint', True))

    model = create_simple_model(input_data = fprints, layer_config = layers_config, output_size = 1)    # get the model

    # train the model using k-fold cross-validation
    best_model, avg_val_rmse, test_rmse, all_train_rmses, all_val_rmses, best_model_scaler_x, best_model_scaler_y = train_model_kfold(
        fprints,  # fingerprints
        temps,    # temperatures
        model,    # the model we made earlier
        criterion=kwargs.get('criterion', None),
        optimiser_class=kwargs.get('optimiser_class', None), # defaults to Adam optimiser
        test_split_size=kwargs.get('test_split_size', 0.1), # these are the holdout test set for the final validation outside the training
        fold_type = fold_type,
        n_repeats = kwargs.get('n_repeats',3), # the number of repeats for repeated k-fold splitting.
        init_type=kwargs.get('init_type', 'xavier_uniform'), # type of initialization to use
        num_epochs=kwargs.get('num_epochs', 500), # number of epochs
        k=kwargs.get('k', 5), # number of folds
        batch_size=kwargs.get('batch_size', 64),
        learn_rate=kwargs.get('learn_rate', 0.001),
        weight_decay=kwargs.get('weight_decay', 1e-5),
        style=kwargs.get('style', 'scatter'), # plot style; hexbin or scatter
        print_output=kwargs.get('print_output', False), # False = minimal output; True = more verbose
        print_output_frequency = kwargs.get('print_output_frequency', 0.2), # the frequency with which to report on epoch training.
        plot_output=kwargs.get('plot_output', False), # to plot or not to plot (obvious I guess)
        limits=kwargs.get('limits', [300,500]), # the limits of the X=Y plot
        patience=kwargs.get('patience', None),
    )

    return best_model, avg_val_rmse, test_rmse, all_train_rmses, all_val_rmses, best_model_scaler_x, best_model_scaler_y

def write_model(model, fp_type, name = '', scaler_x = None, scaler_y = None):
    '''
    saves a model with a structured filename based on fp_type
    notes:
        Apparently this is not the way to do this... https://wandb.ai/wandb/common-ml-errors/reports/How-to-Save-and-Load-Models-in-PyTorch--VmlldzozMjg0MTE
        the scaler-y is needed for getting "real number" predictions later.

    args:
        model: the torch model to be saved.
        fp_type: the fingerprint type to encode in the filename; no point saving if we've got no idea what this is.
        name: an (optional) additional name string to append to the filename.
        scaler_y: the scaler used for output normalization (optional).

    returns:
        Nowt; just saves

    example:
        neurnet.write_model(best_model, fp_type='mordred_fingerprint_3D', name='test', scaler_y=scaler)
        >> Model saved as: mordred_fingerprint_3D_test.pth
    '''

    if not name.endswith('.pth'):
        name += '.pth'

    filename = f"{fp_type}_{name}"

    save_dict = {'model': model, 'scaler_x': scaler_x, 'scaler_y': scaler_y}

    torch.save(save_dict, filename)

    print(f"Model and scaler saved as: {filename}")

def reload_model(model, filename):
    '''
    Reload the model's state dict from the saved file.

    args:
        model: the torch model instance to load the weights into.
        filename: The file containing the saved model weights.

    returns:
        model: The model with loaded weights.

    example:
        neurnet.reload_model(new_model,filename ='mordred_fingerprint_3D_test.pth')
        >> Model loaded from: {filename}
    '''

    if not os.path.exists(filename):
        raise FileNotFoundError(f"No file found at {filename}")

    loaded_dict = torch.load(filename, weights_only = False)

    model = loaded_dict['model']
    scaler_x = loaded_dict.get('scaler_x', None)
    scaler_y = loaded_dict.get('scaler_y', None)

    print(f"Model loaded from: {filename}")

    return model, scaler_x, scaler_y
