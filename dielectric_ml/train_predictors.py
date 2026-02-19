"""
training functions for training vae, classifier and predictor 
"""

import torch
import os

from datetime import datetime

from dielectric_ml import data, engine, utils, models
from itertools import product

from torch.utils.tensorboard import SummaryWriter


def create_classifier_configs(model, n_conv_layers, dropouts, h_dim1s, learning_rates, batch_sizes, epochs, n_splits, n_repeats, prefix=None):
    configs = []
    for n_conv_layer, dropout, h_dim, lr, batch_size, epoch, n_split, n_repeat in product(
        n_conv_layers, dropouts, h_dim1s, learning_rates, batch_sizes, epochs, n_splits, n_repeats
    ):
        # Include prefix in writer_name if provided
        writer_name = f"{model.__name__}_n_{n_conv_layer}_d{dropout}_hdim_{h_dim}_lr_{lr}_bs_{batch_size}_e_{epoch}_s_{n_split}_r_{n_repeat}"
        
        
        config = {
            "loss_fn": torch.nn.BCELoss(),
            "lr": lr,
            "n_conv_layers": n_conv_layer,
            "h_dim": h_dim,
            "dropout": dropout,
            "batch_size": batch_size,
            "epochs": epoch,
            "n_splits": n_split,
            "n_repeats": n_repeat,
            "log_dir": "runs",
            "writer_name": writer_name,
            "prefix": prefix  # Store prefix in config
        }
        configs.append(config)
    return configs

def create_predictor_configs(model, n_conv_layers, dropouts, h_dims, learning_rates, batch_sizes, epochs, n_splits, prefix=None):
    configs = []
    for n_conv_layer, dropout, h_dim, lr, batch_size, epoch, n_split in product(
        n_conv_layers, dropouts, h_dims, learning_rates, batch_sizes, epochs, n_splits
    ):
        writer_name = f"{model.__name__}_n_{n_conv_layer}_d{dropout}_hdim_{h_dim}_lr_{lr}_bs_{batch_size}_e_{epoch}_s_{n_split}"
        
        
        config = {
            "loss_fn": torch.nn.MSELoss(),
            "n_conv_layer": n_conv_layer,
            "h_dim": h_dim,
            "dropout": dropout,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epoch,
            "n_splits": n_split,
            "log_dir": "runs",
            "writer_name": writer_name,
            "prefix": prefix  # Store prefix in config for consistency
        }
        configs.append(config)
    return configs

def create_vae_configs(h_dim1s, h_dim2s, z_dims, learning_rates, out_dim, epochs, n_splits, batch_sizes, prefix=None):
    """create configuration dictionaries for the hyperparameter grid search"""
    configs = []
    for h_dim1, h_dim2, z_dim, lr, epoch, n_split, batch_size in product(
        h_dim1s, h_dim2s, z_dims, learning_rates, epochs, n_splits, batch_sizes
    ):
        # Include prefix in writer_name if provided
        writer_name = f"VAE_lr_{lr}_hdim1_{h_dim1}_hdim2_{h_dim2}_zdim_{z_dim}_bs_{batch_size}_e_{epoch}_s_{n_split}"
        
        
        config = {
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epoch,
            "n_splits": n_split,
            "h_dim1": h_dim1,
            "h_dim2": h_dim2,
            "z_dim": z_dim,
            "out_dim": out_dim,
            "log_dir": "runs",
            "writer_name": writer_name,
            "prefix": prefix  # Store prefix in config
        }
        configs.append(config)
    return configs

# loop through config dict saving each model training loop
def run_vae_experiment(config, model_type: str, model_architecture, data_list, device, prefix=None):
    # Use prefix from config if not provided as parameter
    if prefix is None:
        prefix = config.get("prefix")
    
    # create directory for each experiment
    log_dir = os.path.join(config["log_dir"], prefix+"_"+model_type, config['writer_name'])
    writer = SummaryWriter(log_dir=log_dir)

    model = model_architecture(
        x_dim=config["out_dim"],
        h_dim1=config["h_dim1"],
        h_dim2=config["h_dim2"],
        z_dim=config["z_dim"]
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    print(f"running experiment: {config['writer_name']}")

    try:
        result = engine.vae_train_repeated_kfold(
            model=model,
            data_list=data_list,
            optimizer=optimizer,
            loss_fn=engine.vae_loss,
            epochs=config["epochs"],
            out_dim=config['out_dim'],
            writer=writer,
            batch_size=config["batch_size"],
            device=device,
            n_splits=config["n_splits"]
        )

        model_save_path = f"{config['writer_name']}.pth"
        # Include prefix in target_dir if provided
        if prefix:
            target_dir = f"models/{prefix}_{model_architecture.__name__}/"
        else:
            target_dir = f"models/{model_architecture.__name__}/"
        
        utils.save_model(model=model, target_dir=target_dir, model_name=model_save_path)
        
    except Exception as e:
        print(f"Error in experiment {config['writer_name']}: {str(e)}")
        result = {"error": str(e)}

    finally:
        writer.close()

    return config['writer_name'], result

def run_classifier_experiment(config, model_type, model_architecture, data_list, device, prefix=None):
    # Use prefix from config if not provided as parameter
    if prefix is None:
        prefix = config.get("prefix")
    
    # create directory for each experiment
    log_dir = os.path.join(config["log_dir"], prefix+"_"+model_type, config['writer_name'])
    writer = SummaryWriter(log_dir=log_dir)

    model = model_architecture(
        x_dim=79,
        h_dim=config['h_dim'],
        n_conv_blocks=config['n_conv_layers'],
        dropout=config['dropout'],
    )
    print(model)
    model.to(device)

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    writer_name = config['writer_name']
    print(f"running experiment: {writer_name}")
    
    try:
        results = engine.classifier_train_repeated_strat(
            model=model,
            data_list=data_list,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=config["epochs"],
            writer=writer,
            batch_size=config["batch_size"],
            n_repeats=config["n_repeats"],
            device=device,
            n_splits=config["n_splits"]
        )
        
        # Include prefix in target_dir if provided
        if prefix:
            target_dir = f"models/{prefix}_{model_architecture.__name__}/"
        else:
            target_dir = f"models/{model_architecture.__name__}/"
        
        utils.save_model(model=model, target_dir=target_dir, model_name=(writer_name + ".pth"))
        
    except Exception as e:
        print(f"Error in experiment {writer_name}: {str(e)}")
        results = {"error": str(e)}
    
    finally:
        writer.close()

    return writer_name, results

def run_predictor_experiment(config, model_type, model_architecture, data_list, prop_scaler, device, prefix=None):
    # create directory for each experiment
    if prefix is None:
        prefix = config.get("prefix")

    log_dir = os.path.join(config["log_dir"], prefix+"_"+model_type, config['writer_name'])
    writer = SummaryWriter(log_dir=log_dir)

    model = model_architecture(
        x_dim=79,
        h_dim=config['h_dim'],
        n_conv_blocks=config['n_conv_layer'],
        dropout=config['dropout'],
    )
    
    print(model)
    model.to(device)
    
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=0.0001)

    writer_name = config['writer_name']
    print(f"running experiment: {config['writer_name']}")

    try:
        results = engine.pred_train_split(
            model=model,
            data_list=data_list,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=config["epochs"],
            writer=writer,
            scaler=prop_scaler,
            batch_size=config["batch_size"],
            device=device,
            test_size=0.1,
            n_splits=config["n_splits"]
        )
        target_dir = f"models/{prefix}_{model_architecture.__name__}/"
        utils.save_model(model=model, target_dir=target_dir, model_name=(config["writer_name"]+".pth"))

    except Exception as e:
        print(f"Error in experiment {writer_name}: {str(e)}")
        results = {"error": str(e)}
    
    finally:
        writer.close()

    return config["writer_name"], results

if __name__ == "__main__":
    # training a dielectric predictor 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = data.process_exp_data("data/new_DE_data.xlsx", prop_col="del_e")
    df = df.dropna(subset="PROP")

    data_list, prop_scaler = data.predictor_dataloader(df, "PROP")

    # set up config dictionary hyper parameters
    n_conv_layers = [3, 5]
    dropouts = [0.15]
    h_dims = [256, 512]
    learning_rates = [0.0005]
    batch_sizes = [64]
    epochs = [500]
    n_splits = [5]
    
    
    model_architectures = [models.Predictor, models.EnhancedPredictor, models.GINPredictor, models.GCNPredictor,
                           models.GatedGraphPredictor, models.TransformerPredictor]
    
    model_architectures = [models.Predictor]

    prefix = "del_e"

    for model_architecture in model_architectures:
        # Can now add optional prefix for classifier experiments
        configs = create_predictor_configs(
            model_architecture, 
            n_conv_layers, 
            dropouts, 
            h_dims, 
            learning_rates, 
            batch_sizes, 
            epochs, 
            n_splits, 
            prefix=prefix  # Optional: add prefix for this experiment
        )
        
        print(f"total configs to run {len(configs)}")

        # Directory will now be created with prefix if provided
        os.makedirs(f"data/{prefix}_data", exist_ok=True)
        
        results = {
              "batch_start_time": datetime.now().isoformat(),
             "batch_start_timestamp": datetime.now().timestamp()
            }

        for i, config in enumerate(configs, 1):
            # run_classifier_experiment will use prefix from config
            writer_name, result = run_predictor_experiment(
                config=config, 
                model_type="Del_E_Predictor", 
                model_architecture=model_architecture, 
                data_list=data_list, 
                prop_scaler=prop_scaler,
                device=device
            )
            results[writer_name] = result
            utils.save_pickle(results=results, pickle_file=f"data/{prefix}_data/{model_architecture.__name__}_actuals_experiment.pkl")
            print(f"Results saved. Progress: {i}/{len(configs)}")