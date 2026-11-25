import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from matplotlib.lines import Line2D

def mkdir_fun(path):
    if not os.path.exists(path):
        os.makedirs(path)


# def scatter_plot(args, targets_preds, title, fig_name, label=None, dpi=20, scatter_color='#0077b6', 
#                  title_add=False, title_res='', x_range_status=None):

def scatter_plot(gt_tensor, pred_tensor, path, title, label=None, dpi=20, scatter_color='#0077b6', title_add=False): 
    """
    Create a scatter plot of x_ut vs y_ut, compute Spearman correlation and R^2,
    and save the plot. The x- and y-axis ranges are determined by the data's min and max.
    """

    columns = ["GT_Tensor", "PRED_Tensor"] 
    gt_tensor = gt_tensor.flatten() 
    pred_tensor = pred_tensor.flatten() 
    data_ut = np.column_stack((gt_tensor, pred_tensor))
    df_ut = pd.DataFrame(data_ut, columns=columns)

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(df_ut["GT_Tensor"], df_ut["PRED_Tensor"], 
                         s=50, c=scatter_color, alpha=0.6)

    # Determine axis limits based on data
    x_min, x_max = gt_tensor.min(), gt_tensor.max()

    x_min, x_max = x_min-0.05, x_max+0.05
    # x_min, x_max = 0, x_ut.max()
    y_min, y_max = pred_tensor.min(), pred_tensor.max()

    # Plot the diagonal line y = x from the lowest to the highest across x & y
    # lowest_val = min(x_min, y_min)*0.95
    # highest_val = max(x_max, y_max)*0.95
    # line_ideal, = ax.plot([lowest_val, highest_val], 
    #                       [lowest_val, highest_val], 
    #                       color='black', linestyle='--', label='Ideal Fit (y = x)')

    # # Fit linear regression and plot the regression line
    # lm = LinearRegression()  
    # lm.fit(x_ut.reshape(-1, 1), y_ut.reshape(-1, 1))
    # x_range = np.linspace(x_min, x_max, 100)
    # y_range = lm.predict(x_range.reshape(-1, 1)) 
    # ax.plot(x_range, y_range, color='red', label='Linear Reg.')

    # Compute Spearman correlation and R² score
    spearman_corr, _ = spearmanr(gt_tensor, pred_tensor)   
    r2 = r2_score(gt_tensor, pred_tensor) 

    # Round for display
    spearman_corr = round(spearman_corr, 2)
    r2 = round(r2, 2)

    # Custom legend entry for Spearman correlation and R²
    if title_add:
        empty_handle = Line2D(
            [], [], linestyle='none', 
            label= title + f'\n     Spearman Corr.: {spearman_corr}      R²: {r2}'
        )

    # Set labels with increased font size
    ax.set_xlabel("Ground Truth", fontsize=50)
    ax.set_ylabel("Predicted Values", fontsize=50)

    # Set tick labels to be thicker
    transparent_color = (0, 0, 0, 0.5)  
    ax.xaxis.set_tick_params(width=10, size=15, labelsize=35, color=transparent_color)
    ax.yaxis.set_tick_params(width=10, size=15, labelsize=35, color=transparent_color)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 

    # Set the actual limits for x and y axes
    if x_min<=0: 
        ax.set_xlim([x_min*1.05, x_max*1.05])
        ax.set_ylim([y_min*1.05, y_max*1.05])
    else:
        ax.set_xlim([x_min*0.9, x_max*1.05])
        ax.set_ylim([y_min*0.9, y_max*1.05])

    # Add the custom legend
    if title_add:
        handles = [empty_handle]
        labels = [h.get_label() for h in handles]
        ax.legend(handles=handles, labels=labels, loc='upper left', handlelength=0,
                  handletextpad=0, fontsize=30, frameon=False, 
                  bbox_to_anchor=(0, 1.12))

    plt.tight_layout(pad=0)
    png_path = path
    plt.savefig(f"{png_path}_{label}.png", dpi=dpi)
#     plt.savefig(f"{png_path}_{label}.pdf", dpi=dpi)
    plt.close()

    return spearman_corr, r2

# def scatter_plot(args, targets_preds, title, fig_name, label=None, dpi=20, scatter_color='#0077b6', 
#                  title_add=False, title_res='', x_range_status=None):

#     all_psi_true_ut = targets_preds[0]    
#     all_psi_hat_ut = targets_preds[1]

#     columns = ["GT_PSI", "PRED_PSI"] 
#     x_ut = all_psi_true_ut.cpu().flatten().numpy()
#     y_ut = all_psi_hat_ut.cpu().flatten().numpy()
#     data_ut = np.column_stack((x_ut, y_ut))
#     df_ut = pd.DataFrame(data_ut, columns=columns)

#     fig, ax = plt.subplots(figsize=(12, 12))
#     scatter = ax.scatter(df_ut["GT_PSI"], df_ut["PRED_PSI"], s=50, c=scatter_color, alpha=0.3)

#     # Fit linear regression and plot the regression line
#     lm = LinearRegression()
#     lm.fit(x_ut.reshape(-1, 1), y_ut.reshape(-1, 1))

#     if x_range_status is None:
#         # Plot the diagonal line y = x (ideal fit)
#         line_ideal, = ax.plot([-0.01, 1.01], [-0.01, 1.01], color='black', 
#                               linestyle='--', label='Ideal Fit (y = x)')
        
#         # Prepare points for regression line
#         x_range = np.linspace(-0.05, 1.05, 100)
#         y_range = lm.predict(x_range.reshape(-1, 1))  

#         # Plot the regression line in blue
#         line_lm, = ax.plot(x_range, y_range, color='blue', label='Linear Fit')

#     else:
#         low = min(targets_preds[0] + targets_preds[1]).item()
#         high = max(targets_preds[0] + targets_preds[1]).item()
        
#         # Plot the diagonal line y = x (ideal fit)
#         line_ideal, = ax.plot([low, high], [low, high], color='black', 
#                               linestyle='--', label='Ideal Fit (y = x)')
        
#         # Prepare points for regression line
#         x_range = np.linspace(low, high, 100)
#         y_range = lm.predict(x_range.reshape(-1, 1))

#         # Plot the regression line in blue
#         line_lm, = ax.plot(x_range, y_range, color='blue', label='Linear Fit')

#     # Compute Spearman correlation and R² score
#     spearman_corr, _ = spearmanr(x_ut, y_ut)   
#     r2 = r2_score(x_ut, y_ut) 
#     spearman_corr = round(spearman_corr, 2)
#     r2 = round(r2, 2)

#     # If you want to show correlation, R², etc. in the legend:
#     if title_add: 
#         # Create a custom legend entry for Spearman correlation
#         empty_handle = Line2D([], [], linestyle='none',
#                               label=f'{title_res}\nSpearman: {spearman_corr}')

#     ax.set_xlabel("Ground Truth", fontsize=50)
#     ax.set_ylabel("Predicted Values", fontsize=50)

#     # Thicker ticks
#     transparent_color = (0, 0, 0, 0.5)  
#     ax.xaxis.set_tick_params(width=10, size=15, labelsize=35, color=transparent_color)
#     ax.yaxis.set_tick_params(width=10, size=15, labelsize=35, color=transparent_color)

#     # Remove top and right spines
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False) 
 
#     # Add the custom legend 
#     if title_add:
#         if x_range_status is None:
#             plt.xlim([-0.05, 0.99])
#             plt.ylim([-0.05, 0.99]) 
        
#         handles = [empty_handle]
#         labels = [h.get_label() for h in handles]
#         ax.legend(handles=handles, labels=labels, loc='upper left', handlelength=0, 
#                   handletextpad=0, fontsize=40, frameon=False, bbox_to_anchor=(0, 1.12))

#     plt.tight_layout(pad=0) 
#     plt.savefig(fig_name, dpi=dpi)  
#     plt.close()

#     return spearman_corr, r2




# import numpy as np 
# import pandas as pd   
# from scipy import stats
# import matplotlib.cm as cm 
# import matplotlib.pyplot as plt 
# from sklearn.metrics import r2_score
# from sklearn.linear_model import LinearRegression 
 
# def scatter_plot(model, targets_preds, fig_name, dpi=200): 
     
#         all_psi_true_ut = targets_preds[0]   
#         all_psi_hat_t = targets_preds[1]  
#         all_psi_hat_ut = model.untransform_targets(all_psi_hat_t)

#         columns = ["GT_PSI", "PRED_PSI"] 
#         x_ut = all_psi_true_ut.cpu().flatten().numpy().tolist()
#         y_ut = all_psi_hat_ut.cpu().flatten().numpy().tolist()
#         data_ut = [[i, j] for (i, j) in zip(x_ut, y_ut)]
#         df_ut = pd.DataFrame(data_ut, columns=columns)

#         fig, ax = plt.subplots(figsize=(12, 8))
#         plt.scatter(df_ut["GT_PSI"], df_ut["PRED_PSI"], s=15, c="black", alpha=0.4) 
#         ax.set_xlabel("Truth")
#         ax.set_ylabel("Predicted")
#         # plt.xlim([-0.1, 1.1])
#         # plt.ylim([-0.1, 1.1])

#         x_ut = all_psi_true_ut.cpu().flatten().numpy() 
#         y_ut = all_psi_hat_ut.cpu().flatten().numpy() 

#         spearman_corr, _ = stats.spearmanr(x_ut, y_ut)  
#         spearman_corr = round(spearman_corr, 3)

#         lm = LinearRegression()  
#         lm.fit(x_ut.reshape(-1, 1), y_ut.reshape(-1, 1)) 
#         y_pred = lm.predict(x_ut.reshape(-1, 1)) 
#         r2 = r2_score(y_ut.reshape(-1, 1), y_pred)
#         r2 = round(np.mean(r2), 3) 

#         plt.text(0.05, 0.95, 'spearman corr: ' + str(spearman_corr) +'\nr2: ' + str(r2), 
#                 fontsize=15,
#                 horizontalalignment='left',
#                 verticalalignment='top',
#                 transform=plt.gca().transAxes,
#                 bbox=dict(facecolor='yellow', alpha=0.2))  # yellow highlight

#         plt.savefig(fig_name, dpi=dpi)
#         plt.close()  

     
     
