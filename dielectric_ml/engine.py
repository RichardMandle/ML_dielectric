"""
* train and test steps for molecule generation varational autoencoder 
* train and test steps for phase classifier
* train and test steps for temperature predictors
"""
import torch 
import torch.nn.functional as F 
import torch.utils.tensorboard
from tqdm import tqdm
from sklearn.model_selection import KFold, RepeatedKFold ,StratifiedKFold, RepeatedStratifiedKFold, train_test_split

from torch_geometric.loader import DataLoader
import numpy as np

from sklearn.metrics import precision_score

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

def vae_loss(recon_x, x, mu, log_var, out_dim):
    """
    vae loss function with kld and bce terms
    """
    
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, out_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD 
    

def vae_train_step(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   loss_fn,
                   device: torch.device,
                   out_dim: int
                   ):
    """
    Trains the vae model for a single epoch

    returns the train loss 
    """
    model.train()
    
    train_loss = 0
    
    for batch_idx, graph in enumerate(dataloader):
        graph = graph.to(device)
        
        
        # pass graph to model
        recon_batch, mu, log_var = model(graph, out_dim)

        # calculate loss
        loss = loss_fn(recon_batch, graph, mu, log_var, out_dim)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # batch metrics
        train_loss += loss.item()

    return train_loss / len(dataloader)

def vae_test_step(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   loss_fn,
                   device: torch.device,
                   out_dim: int
                   ):
    """
    tests the vae model for a single epoch
    """
    model.eval()
    
    test_loss = 0

    with torch.no_grad():
        for batch_idx, graph in enumerate(dataloader):
            graph = graph.to(device)
            
            # pass graph
            recon_batch, mu, log_var = model(graph, out_dim)
            
            # calculate loss 
            loss = loss_fn(recon_batch, graph, mu, log_var, out_dim)
            
            #batch metrics
            test_loss += loss.item()

        return test_loss / len(dataloader)

def vae_train_kfold(
        model: torch.nn.Module,
        data_list: list,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        epochs: int,
        device: torch.device,
        out_dim: int,
        writer: torch.utils.tensorboard.SummaryWriter,
        batch_size: int = 32,
        lr: int = 0.001,
        n_splits: int = 5,
        kfold: bool = True 
):
    """
    tests and trains classifer with option for k fold splitting
    can save run with summary writer

    returns
    results: dict of loss and acc
    """
    kf = KFold(n_splits=n_splits, shuffle = True)
    logging_interval = epochs // 10 
        # eval metrics
    results = {
        "train_loss": [],
        "test_loss": [],
    }

    model.apply(lambda m: m.reset_parameters() if hasattr (m, 'reset_parameters') else None)
    # to use kfold splitting or not. add kfold option
    if kfold:
        for fold, (train_idx, test_idx) in enumerate(kf.split(data_list)):
            print(f"starting fold {fold + 1} / {n_splits}...")

            # use kfold indexing to create data lists
            train_dataset = [data_list[i] for i in train_idx]
            test_dataset = [data_list[i] for i in test_idx]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


            for epoch in tqdm(range(epochs)):
                avg_train_loss = vae_train_step(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device,
                    out_dim=out_dim
                )

                # test step
                avg_test_loss = vae_test_step(
                    model = model,
                    dataloader= test_loader,
                    loss_fn=loss_fn,
                    device=device,
                    out_dim=out_dim
                )

                # print metrics 
                if epoch % logging_interval == 0:
                    print(f"Fold: {fold+1}, Epoch: {epoch}/{epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Test Loss: {avg_test_loss:.4f}")

                writer.add_scalars("Loss", {
                    "Train": avg_train_loss,
                    "Test": avg_test_loss
                }, epoch)
               
                # store fold results
                results['train_loss'].append(avg_train_loss)
                results['test_loss'].append(avg_test_loss)

            # end of fold close writer
            print(f"completed fold {fold+1}/{n_splits}")
 

        print("Training completed")
        return results

def vae_train_repeated_kfold(
        model: torch.nn.Module,
        data_list: list,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        epochs: int,
        device: torch.device,
        out_dim: int,
        writer: torch.utils.tensorboard.SummaryWriter,
        batch_size: int = 32,
        n_repeats=3,
        lr: int = 0.001,
        n_splits: int = 5,
        kfold: bool = True 
):
    """
    tests and trains classifer with option for k fold splitting
    can save run with summary writer

    returns
    results: dict of loss and acc
    """
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    logging_interval = epochs // 10 
        # eval metrics
    results = {
        "train_loss": [],
        "test_loss": [],
    }

    model.apply(lambda m: m.reset_parameters() if hasattr (m, 'reset_parameters') else None)
    
    # to use kfold splitting or not. add kfold option
    if kfold:
        global_step = 0
        for fold, (train_idx, test_idx) in enumerate(kf.split(data_list)):
            print(f"starting fold {fold + 1} / {n_splits * n_repeats}...")

            # use kfold indexing to create data lists
            train_dataset = [data_list[i] for i in train_idx]
            test_dataset = [data_list[i] for i in test_idx]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


            for epoch in tqdm(range(epochs)):
                avg_train_loss = vae_train_step(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device,
                    out_dim=out_dim
                )

                # test step
                avg_test_loss = vae_test_step(
                    model = model,
                    dataloader= test_loader,
                    loss_fn=loss_fn,
                    device=device,
                    out_dim=out_dim
                )

                # print metrics 
                if epoch % logging_interval == 0:
                    print(f"Fold: {fold+1}, Epoch: {epoch}/{epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Test Loss: {avg_test_loss:.4f}")

                writer.add_scalars("Loss", {
                    "Train": avg_train_loss,
                    "Test": avg_test_loss
                }, global_step)
               
                # store fold results
                results['train_loss'].append(avg_train_loss)
                results['test_loss'].append(avg_test_loss)

                global_step += 1

            # end of fold close writer
            print(f"completed fold {fold+1}/{n_splits}")
 

        print("Training completed")
        return results

def classifier_train_step(model: torch.nn.Module,
                        dataloader: torch.utils.data.DataLoader,
                        optimizer: torch.optim.Optimizer,
                        loss_fn,
                        device: torch.device):
    """
    train step for classifier 
    """
    model.train()

    total_train_loss, total_train_acc = 0, 0
    all_preds, all_targets = [], []

    for batch_idx, graph in enumerate(dataloader):
        graph = graph.to(device)
    
        # pass graph to model
        y_pred = model(graph.x, graph.edge_index, graph.batch)
        pred = y_pred.round() 

        loss = loss_fn(y_pred, graph.prop)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # batch metrics
        total_train_loss += loss.item()
        total_train_acc += (y_pred.round() == graph.prop).float().mean().item()

        all_preds.extend(pred.detach().cpu().numpy())
        all_targets.extend(graph.prop.cpu().numpy())


    # get average loss per batch
    avg_train_loss = total_train_loss / len(dataloader)
    avg_train_acc = total_train_acc / len(dataloader)
    avg_precision = precision_score(all_targets, all_preds, average="binary", zero_division=np.nan)

    return avg_train_loss, avg_train_acc, avg_precision


def classifier_test_step(model: torch.nn.Module,
                        dataloader: torch.utils.data.DataLoader,
                        loss_fn,
                        device: torch.device):
    """
    test step for classifier
    """
    model.eval()

    total_test_loss, total_test_acc = 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_idx, graph in enumerate(dataloader):
            graph = graph.to(device)
        
            # forward pass
            y_pred = model(graph.x, graph.edge_index, graph.batch)
            pred = y_pred.round()
            loss = loss_fn(y_pred, graph.prop)
            
            # batch metrics
            total_test_loss += loss.item()
            total_test_acc += (y_pred.round() == graph.prop).float().mean().item()

        all_preds.extend(pred.detach().cpu().numpy())
        all_targets.extend(graph.prop.cpu().numpy())

        # get average loss per batch
        avg_train_loss = total_test_loss / len(dataloader)
        avg_train_acc = total_test_acc / len(dataloader)
        avg_precision = precision_score(all_targets, all_preds, average="binary", zero_division=np.nan)


        return avg_train_loss, avg_train_acc, avg_precision

def classifier_train_strat(
    model: torch.nn.Module,
    data_list: list,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    epochs: int,
    device: torch.device,
    writer:torch.utils.tensorboard.SummaryWriter,
    batch_size: int = 32,
    n_splits: int = 5,
    kfold: bool = True
    
):
    """
    tests and trains classifer with option for k fold splitting
    can save run with summary writer

    returns
    results: dict of loss and acc
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle = True)
    logging_interval = epochs // 10 
    results = {
                "train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": []
            }

    target_labels = [data.prop.item() for data in data_list]

    model.apply(lambda m: m.reset_parameters() if hasattr (m, 'reset_parameters') else None)

    # to use kfold splitting or not. add kfold option
    if kfold:
        for fold, (train_idx, test_idx) in enumerate(kf.split(data_list, target_labels)):
            print(f"starting fold {fold + 1} / {n_splits}...")


            # use kfold indexing to create data lists
            train_dataset = [data_list[i] for i in train_idx]
            test_dataset = [data_list[i] for i in test_idx]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            example_batch = next(iter(train_loader))
            example_batch = example_batch.to(device)

            

            for epoch in tqdm(range(epochs)):
                avg_train_loss, avg_train_acc = classifier_train_step(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device
                )

                # test step
                avg_test_loss, avg_test_acc = classifier_test_step(
                    model = model,
                    dataloader= test_loader,
                    loss_fn=loss_fn,
                    device=device
                )

                # print metrics 
                if epoch % logging_interval == 0:
                    print(f"Fold: {fold+1}, Epoch: {epoch}/{epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
                        f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")

                # experiment tracking
                writer.add_scalars("Loss", {
                    "Train": avg_train_loss,
                    "Test": avg_test_loss
                }, epoch)

                writer.add_scalars("Accuracy",{
                    "Train": avg_train_acc,
                    "Test": avg_test_acc
                }, epoch)

                if fold == 0 and epoch == 0:
                    writer.add_graph(model, [example_batch.x, example_batch.edge_index, example_batch.batch])

                # store fold results
                results['train_loss'].append(avg_train_loss)
                results['test_loss'].append(avg_test_loss)
                results['train_acc'].append(avg_train_acc)
                results['test_acc'].append(avg_test_acc)

            # end of fold close writer
            print(f"completed fold {fold+1}/{n_splits}")

        
        print("Training completed")
        return results

def classifier_train_repeated_strat(
    model: torch.nn.Module,
    data_list: list,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    epochs: int,
    device: torch.device,
    writer:torch.utils.tensorboard.SummaryWriter,
    batch_size: int = 32,
    n_splits: int = 5,
    n_repeats:int = 3,
    kfold: bool = True

):
    """
    tests and trains classifer with option for k fold splitting
    can save run with summary writer

    returns
    results: dict of loss and acc
    """
    kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    logging_interval = epochs // 10 
    results = {
                "train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": [],
                "train_precision": [],
                "test_precision": []
            }

    target_labels = [data.prop.item() for data in data_list]

    model.apply(lambda m: m.reset_parameters() if hasattr (m, 'reset_parameters') else None)

    # to use kfold splitting or not. add kfold option
    if kfold:
        global_step = 0
        for fold, (train_idx, test_idx) in enumerate(kf.split(data_list, target_labels)):
            print(f"starting fold {fold + 1} / {n_splits * n_repeats}...")


            # use kfold indexing to create data lists
            train_dataset = [data_list[i] for i in train_idx]
            test_dataset = [data_list[i] for i in test_idx]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            example_batch = next(iter(train_loader))
            example_batch = example_batch.to(device)

            # reset parameters each fold for assessing generalisability 
            #model.apply(lambda m: m.reset_parameters() if hasattr (m, 'reset_parameters') else None)

            for epoch in tqdm(range(epochs)):
                avg_train_loss, avg_train_acc, avg_train_precision = classifier_train_step(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device
                )

                # test step
                avg_test_loss, avg_test_acc, avg_test_precision = classifier_test_step(
                    model = model,
                    dataloader= test_loader,
                    loss_fn=loss_fn,
                    device=device
                )

                # print metrics 
                if epoch % logging_interval == 0:
                    print(f"Fold: {fold+1}, Epoch: {epoch}/{epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Train Precision: {avg_train_precision:.4f}"
                        f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}, Test Precision: {avg_test_precision:.4f}")

                # experiment tracking
                writer.add_scalars("Loss", {
                    "Train": avg_train_loss,
                    "Test": avg_test_loss
                }, global_step)

                writer.add_scalars("Accuracy",{
                    "Train": avg_train_acc,
                    "Test": avg_test_acc
                }, global_step)

                writer.add_scalars("Precision", {
                    "Train": avg_train_precision,
                    "Test": avg_test_precision
                }, global_step)

                if fold == 0 and epoch == 0:
                    writer.add_graph(model, [example_batch.x, example_batch.edge_index, example_batch.batch])

                # store fold results
                results['train_loss'].append(avg_train_loss)
                results['test_loss'].append(avg_test_loss)
                results["train_precision"].append(avg_train_precision)
                results["test_precision"].append(avg_test_precision)
                results['train_acc'].append(avg_train_acc)
                results['test_acc'].append(avg_test_acc)

                global_step += 1

            # end of fold close writer
            print(f"completed fold {fold+1}/{n_splits}")
        
        
        print("Training completed")
        return results

def predict(model, dataloader, device):
    """
    predict values
    """
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_idx, graph in enumerate(dataloader):
            graph = graph.to(device)
            pred = model(graph.x, graph.edge_index, graph.batch)
            actual = graph.norm_prop
            
            predictions.append(pred.cpu().numpy())
            actuals.append(actual.cpu().numpy())

            return np.concatenate(predictions), np.concatenate(actuals)

def get_rmse(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    scaler,

    device
):
    """
    computes the rmse for a model
    """
    preds, acts = predict(model, loader,device)
    #preds = preds.reshape(-1, 1).flatten()
    #acts = acts.reshape(-1, 1).flatten()
    

    p_unscaled = scaler.inverse_transform(preds)
    a_unscaled = scaler.inverse_transform(acts)
    rmse = np.sqrt(np.mean((p_unscaled - a_unscaled) ** 2))
    
    return rmse

def pred_train_step(model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    loss_fn,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device):
    model.train()
    
    train_loss = 0
    
    for batch_idx, graph in enumerate(dataloader):
        graph = graph.to(device)

        # pass graph to model
        out = model(graph.x, graph.edge_index, graph.batch)

        # calculate loss
        loss = loss_fn(out, graph.norm_prop)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #batch metrics
        train_loss += loss.item()
    
    return train_loss / len(dataloader)  

def pred_test_step(
                    model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    loss_fn,
                    device: torch.device
):
    """
    tests the prediction model for a single epoch against its norm
    temp value
    """
    model.eval()

    test_loss = 0

    with torch.no_grad():
        for batch_idx, graph in enumerate(dataloader):
            graph = graph.to(device)

            # forward pass
            out = model(graph.x, graph.edge_index, graph.batch)


            # calculate loss
            loss = loss_fn(out, graph.norm_prop)
            
            # batch metrics 
            test_loss += loss.item()

    return test_loss / len(dataloader) 

def pred_train_split(
    model: torch.nn.Module,
    data_list: list,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    epochs: int,
    device: torch.device,
    writer:torch.utils.tensorboard.SummaryWriter,
    scaler, 
    batch_size: int = 32,
    n_splits: int = 5,
    test_size: float = 0.2,
    kfold: bool = True,
):
    """
    tests and trains temperature predictor with option fo

    this version has train/val/test
    """

    # hold out test set
    train_val_data, test_data = train_test_split(
        data_list,
        test_size=test_size
    )

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    kf = KFold(n_splits=n_splits, shuffle=True)
    logging_interval = max(1, epochs // 10)
    results = {
                "train_loss": [],
                "val_loss": [],
                "val_rmse":[],
                "test_loss": None,
                "test_rmse": None
                }

    model.apply(lambda m: m.reset_parameters() if hasattr (m, 'reset_parameters') else None)

    # to use kfold splitting or not. add kfold option
    if kfold:
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_data)):
            print(f"starting fold {fold + 1} / {n_splits}...")

            # use kfold indexing to create data lists
            train_dataset = [train_val_data[i] for i in train_idx]
            val_dataset = [train_val_data[i] for i in val_idx]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            example_batch = next(iter(train_loader))
            example_batch = example_batch.to(device)

            early_stopping = EarlyStopping(patience=30, min_delta=1e-4)

            for epoch in tqdm(range(epochs)):
                avg_train_loss = pred_train_step(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device
                )

                # test step
                avg_val_loss = pred_test_step(
                    model = model,
                    dataloader= val_loader,
                    loss_fn=loss_fn,
                    device=device
                )

                with torch.no_grad():
                    val_rmse = get_rmse(model, val_loader, scaler, device)

                # print metrics 
                if epoch % logging_interval == 0:
                    print(f"fold: {fold+1}, epoch{epoch}/{epochs}"
                          f"train loss: {avg_train_loss:.4f}"
                          f"val loss: {avg_val_loss:.4f}"
                          f"rmse:{val_rmse:.4f}")
        
                # experiment tracking
                writer.add_scalars("Loss", {
                    "Train": avg_train_loss,
                    "Val": avg_val_loss
                }, epoch+ fold * epochs)

                writer.add_scalar("Val_RMSE", val_rmse, epoch+ fold * epochs)

                if fold == 0 and epoch == 0:
                    writer.add_graph(model, [example_batch.x, example_batch.edge_index, example_batch.batch])

                # store fold results
                results['train_loss'].append(avg_train_loss)
                results['val_loss'].append(avg_val_loss)
                results["val_rmse"].append(val_rmse)

                early_stopping(val_rmse,model)
                if early_stopping.should_stop:
                    print(f"Early stopping at epoch{epoch} for fold {fold}")
            # end of fold close writer
            print(f"completed fold {fold+1}/{n_splits}")

        early_stopping.restore_best(model)
        
        print("Evaluating on hold out test set")
        results['test_loss'] = pred_test_step(
            model=model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            device=device
        )
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_actuals = []

            for batch in test_loader:
                if batch.num_graphs == 1:
                    continue
                batch = batch.to(device)
                preds = model(batch.x, batch.edge_index, batch.batch)
                all_preds.append(preds.cpu())
                all_actuals.append(batch.norm_prop.cpu())

        preds_tensor = torch.cat(all_preds).numpy()
        actuals_tensor = torch.cat(all_actuals).numpy()
    
    # Inverse transform to get original scale
        results["test_preds"] = scaler.inverse_transform(preds_tensor.reshape(-1, 1)).flatten()
        results["test_actuals"] = scaler.inverse_transform(actuals_tensor.reshape(-1, 1)).flatten()


        results['test_rmse'] = get_rmse(model, test_loader, scaler, device)

        print(f"final test loss: {results['test_loss']:.4f}")
        print(f"final test rmse: {results['test_rmse']:.4f}")

        print("Training completed")
        return results
    
def pred_train_repeated(
    model: torch.nn.Module,
    data_list: list,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    epochs: int,
    device: torch.device,
    writer:torch.utils.tensorboard.SummaryWriter,
    scaler,
    batch_size: int = 32,
    n_splits: int = 5,
    n_repeats:int = 3,
    kfold: bool = True,
):
    """
    tests and trains temperature predictor with option fo
    """
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    logging_interval = epochs // 10
    results = {
                "train_loss": [],
                "test_loss": [],
                "rmse":[]
                }


    model.apply(lambda m: m.reset_parameters() if hasattr (m, 'reset_parameters') else None)
    
    # to use kfold splitting or not. add kfold option
    if kfold:
        global_step = 0
        for fold, (train_idx, test_idx) in enumerate(kf.split(data_list)):
            print(f"starting fold {fold + 1} / {n_splits * n_repeats}...")

            # use kfold indexing to create data lists
            train_dataset = [data_list[i] for i in train_idx]
            test_dataset = [data_list[i] for i in test_idx]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            example_batch = next(iter(train_loader))
            example_batch = example_batch.to(device)

            

            for epoch in tqdm(range(epochs)):
                avg_train_loss = pred_train_step(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device
                )

                # test step
                avg_test_loss = pred_test_step(
                    model = model,
                    dataloader= test_loader,
                    loss_fn=loss_fn,
                    device=device
                )

                with torch.no_grad():
                    rmse = get_rmse(model, test_loader, scaler, device)

                # print metrics 
                if epoch % logging_interval == 0:
                    print(f"fold: {fold+1}, epoch{epoch}/{epochs}"
                          f"train loss: {avg_train_loss}"
                          f"test loss: {avg_test_loss}"
                          f"rmse:{rmse}")
        
                # experiment tracking
                writer.add_scalars("Loss", {
                    "Train": avg_train_loss,
                    "Test": avg_test_loss
                }, global_step)

                writer.add_scalars("RMSE", {
                    "RMSE":rmse
                }, global_step)

                if fold == 0 and epoch == 0:
                    writer.add_graph(model, [example_batch.x, example_batch.edge_index, example_batch.batch])

                # store fold results
                results['train_loss'].append(avg_train_loss)
                results['test_loss'].append(avg_test_loss)
                results["rmse"].append(rmse)

                global_step += 1

            # end of fold close writer
            print(f"completed fold {fold+1}/{n_splits}")


        print("Training completed")
        return results

