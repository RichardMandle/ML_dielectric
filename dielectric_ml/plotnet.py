# plotnet - plotting stuff

import matplotlib.pyplot as plt
import numpy as np

def plot_model(train_actuals_unscaled, train_preds_unscaled, val_actuals_unscaled, val_preds_unscaled, train_rmse, val_rmse, fold, train_rmses, val_rmses, gridsize=25, cmap='Blues', mincnt=1, plot_xy=True, plot_xy_color='red', style='hexbin', limits = [300,500]):
    '''
    Function for plotting model results; training, validation; RMSE vs epoch for both
    '''
    plt.figure(figsize=(12, 4))
    
    style = style.lower()
    if style != 'hexbin':
        style = 'scatter'
        
    def make_plot(style, x, y, gridsize, cmap, mincnt, plot_xy):
        '''
        Plotting function for consistency.
        '''
        if style == 'hexbin':
            hb = plt.hexbin(x, y, gridsize=gridsize, cmap=cmap, mincnt=mincnt)
            plt.colorbar(hb, label='Counts')
            
        if style == 'scatter':
            hb = plt.scatter(x, y)
        
        plt.xlabel('Actual')
        plt.ylabel('Predicted')

        if plot_xy:
            plt.plot([limits[0], limits[1]], [limits[0], limits[1]], label='X=Y', color=plot_xy_color)
            plt.legend()
        
        return hb
        
    # training plot
    plt.subplot(1, 3, 1)
    hb = make_plot(style = style, x = train_actuals_unscaled, y = train_preds_unscaled, gridsize=gridsize, cmap=cmap, mincnt=mincnt, plot_xy = plot_xy)
    plt.title(f'Training Data Fold {fold}; RMSE: {train_rmse:.4f}')

    # validation plot
    plt.subplot(1, 3, 2)
    hb = make_plot(style = style, x = val_actuals_unscaled, y = val_preds_unscaled, gridsize=gridsize, cmap=cmap, mincnt=mincnt, plot_xy = plot_xy)
    plt.title(f'Validation Data Fold {fold}; RMSE: {val_rmse:.4f}')

    # RMSE over epochs plot
    plt.subplot(1, 3, 3)
    epochs = range(1, len(train_rmses) + 1)
    plt.plot(epochs, train_rmses, label='Train RMSE', color='blue')
    plt.plot(epochs, val_rmses, label='Validation RMSE', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title(f'RMSE Over Epochs Fold {fold}')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def plot_test_results(test_actuals_unscaled, test_preds_unscaled, test_rmse, gridsize=25, cmap='Blues', mincnt=1, plot_xy=True, plot_xy_color='red', style='hexbin', limits = [300,500]):
    '''
    Function for plotting test results; actual vs predicted; RMSE for the test set
    '''
    plt.figure(figsize=(6, 6))
    
    style = style.lower()
    if style != 'hexbin':
        style = 'scatter'
        
    def make_plot(style, x, y, gridsize, cmap, mincnt, plot_xy):
        '''
        Plotting function for consistency.
        '''
        if style == 'hexbin':
            hb = plt.hexbin(x, y, gridsize=gridsize, cmap=cmap, mincnt=mincnt)
            plt.colorbar(hb, label='Counts')
            
        if style == 'scatter':
            hb = plt.scatter(x, y)
        
        plt.xlabel('Actual')
        plt.ylabel('Predicted')

        if plot_xy:
            plt.plot([limits[0], limits[1]], [limits[0], limits[1]], label='X=Y', color=plot_xy_color)
            plt.legend()
        
        return hb
        
    # test plot
    hb = make_plot(style = style, x = test_actuals_unscaled, y = test_preds_unscaled, gridsize=gridsize, cmap=cmap, mincnt=mincnt, plot_xy = plot_xy)
    plt.title(f'Test Data; RMSE: {test_rmse:.4f}')
    
    plt.tight_layout()
    plt.show()
    
def plot_rmse_comparison(analysis_dict):
    """
    plots the comparison of training, validation, and test RMSEs for different fingerprint types on separate subplots.

    args:
    - analysis_dict: dict with analyzed results for each fingerprint type generated via neurnet.analyse_results.
    """
    fp_types = []
    mean_val_rmses = []
    std_val_rmses = []
    test_rmses = []
    mean_train_rmses = []
    std_train_rmses = []

    for fp_type, stats in analysis_dict.items():
        fp_types.append(fp_type)
        mean_val_rmses.append(stats['mean_val_rmse'])
        std_val_rmses.append(stats['std_val_rmse'])
        test_rmses.append(stats['test_rmse'])
        mean_train_rmses.append(stats['mean_train_rmse'])
        std_train_rmses.append(stats['std_train_rmse'])

    mean_val_rmses = np.array(mean_val_rmses)
    std_val_rmses = np.array(std_val_rmses)
    test_rmses = np.array(test_rmses)
    mean_train_rmses = np.array(mean_train_rmses)
    std_train_rmses = np.array(std_train_rmses)

    sorted_indices = np.argsort(mean_val_rmses)[::-1]
    fp_types = np.array(fp_types)[sorted_indices]
    mean_val_rmses = mean_val_rmses[sorted_indices]
    std_val_rmses = std_val_rmses[sorted_indices]
    test_rmses = test_rmses[sorted_indices]
    mean_train_rmses = mean_train_rmses[sorted_indices]
    std_train_rmses = std_train_rmses[sorted_indices]

    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    x = np.arange(len(fp_types))  # the label locations

    # training RMSE subplot
    axes[0].bar(x, mean_train_rmses, yerr=std_train_rmses, capsize=5, color='skyblue')
    axes[0].set_title('Training RMSE across Fingerprint Types')
    axes[0].set_ylabel('Train RMSE')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(fp_types, rotation=45, ha='right')

    # validation RMSE subplot
    axes[1].bar(x, mean_val_rmses, yerr=std_val_rmses, capsize=5, color='orange')
    axes[1].set_title('Validation RMSE across Fingerprint Types')
    axes[1].set_ylabel('Val RMSE')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(fp_types, rotation=45, ha='right')

    # test RMSE subplot
    axes[2].bar(x, test_rmses, color='green')
    axes[2].set_title('Test RMSE across Fingerprint Types')
    axes[2].set_ylabel('Test RMSE')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(fp_types, rotation=45, ha='right')

    fig.tight_layout()
    plt.show()