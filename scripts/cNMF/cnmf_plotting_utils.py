import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import matplotlib as mpl
import numpy as np


def plot_usages_umap(
    adata,
    output_prefix,
    usages,
    n_cols = 5,
    figsize = (20, 15),
    dpi = 600,
    rename_dict = None,  # Dictionary for renaming plots
    ):
    """
    Generates a grid of UMAP plots for specified usage columns in an AnnData object.
    Allows renaming plots using a dictionary.

    Args:
        adata (AnnData): The AnnData object.
        output_prefix (str): Path to save the plot.
        n_cols (int, optional): Number of columns in the subplot grid. Defaults to 5.
        figsize (tuple, optional): Figure size (width, height). Defaults to (20, 15).
        dpi (int, optional): Resolution (dots per inch). Defaults to 800.
        usages (int, optional): Number of usages to plot. Defaults to 25.
        rename_dict (dict, optional): Dictionary mapping original usage names to new names. Defaults to None.
    """

    usage_cols = [f"usage_{i}" for i in range(1, usages + 1)]
    n_rows = (len(usage_cols) + n_cols - 1) // n_cols

    mpl.rcParams['figure.dpi'] = dpi
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['axes.facecolor'] = 'white'

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(usage_cols):
        if col in adata.obs.columns:
            ax = axes[i]
            ax.set_facecolor('white')
            sc.pl.umap(adata, color=col, ax=ax, show=False)

            # Use rename_dict if provided, otherwise use original name
            title = rename_dict.get(col, col) if rename_dict else col
            axes[i].set_title(title)
        else:
            print(f"Warning: Column '{col}' not found in adata.obs")
            axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.set_facecolor('white')
    plt.tight_layout()

    # Create full path
    if rename_dict == None:
        output_filename = output_prefix + '_usage_umap_plots.png'
    else:
        output_filename = output_prefix + '_labelled_usage_umap_plots.png'
        
    plt.savefig(output_filename)
    plt.close(fig)

def plot_celltypes_umap(
    adata,
    output_prefix,
    ncols = 6,
    figsize = (40, 20),
    dpi = 600,
    point_size = 0.5):
    """Generates a combined UMAP plot, with adjustable point size, for all unique cell types.

    Args:
        adata: The AnnData object.
        ncols: Number of columns in the subplot grid.
        output_filename: Filename to save the combined plot.
        figsize: Overall figure dimensions.
        dpi: Resolution in dots per inch.
        point_size: Size of the points in the scatter plot (adjust for visual matching).
    """
    unique_cell_types = adata.obs['Cell_Type'].unique()
    nrows = (len(unique_cell_types) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, facecolor="white")
    axes = axes.ravel()

    for i, cell_type in enumerate(unique_cell_types):
        color_vector = np.where(adata.obs['Cell_Type'].str.strip() == cell_type, 'red', 'lightgray')
        x = adata.obsm['X_umap'][:, 0]
        y = adata.obsm['X_umap'][:, 1]

         # Plot gray points first
        axes[i].scatter(x, y, c='lightgray', alpha = 0.2, marker = 'o', s=point_size, cmap=None)

        # Plot red points on top
        indices_red = np.where(adata.obs['Cell_Type'].str.strip() == cell_type)[0]
        axes[i].scatter(x[indices_red], y[indices_red], c='red', marker = 'o', alpha = 0.6, s=point_size, cmap=None)
        
        #axes[i].scatter(x, y, c=color_vector, alpha = 0.1, s=point_size, cmap=None)  #Use point_size here.
        axes[i].set_title(cell_type)
        axes[i].set_xlabel("UMAP1")
        axes[i].set_ylabel("UMAP2")
        axes[i].set_xticks([]) #Remove tick marks for cleaner look
        axes[i].set_yticks([])

    for ax in axes[len(unique_cell_types):]:
        ax.axis('off')
    fig.set_facecolor('white')
    plt.tight_layout()
    output_filename = output_prefix + '_celltype_umap_plots.png'
    plt.savefig(output_filename, dpi=dpi)
    plt.close(fig)


def plot_spectra_scores(
    spectra_file,
    output_prefix,
    n_genes = 25,
    fig_rows = 5,
    fig_cols = 5,
    save_plot = True,
    fig_height = 24,
    fig_width = 18,
    rename_dict = None,
    dpi = 600):
    """Plots the top genes and their scores for each usage in a spectra file, with optional renaming.

    Args:
        spectra_file (str): Path to the spectra file.
        output_prefix (str): Prefix for the saved plot filename.
        n_genes (int, optional): Number of top genes to plot. Defaults to 25.
        fig_rows (int, optional): Number of rows in the plot grid. Defaults to 5.
        fig_cols (int, optional): Number of columns in the plot grid. Defaults to 5.
        save_plot (bool, optional): Whether to save the plot to a file. Defaults to True.
        rename_dict (dict, optional): A dictionary mapping "usage_i" to new names. Defaults to None.
        dpi (int, optional): DPI for saving the plot. Defaults to 600.
    """
    
    # Load the text file into a pandas DataFrame
    df = pd.read_csv(spectra_file, sep="\t", index_col=0)  
    df = df.transpose()

    # Get the number of replicates (usages)
    num_replicates = len(df.columns)

    # Create a fig_rows x fig_cols grid of subplots
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_width, fig_height))  
    axes = axes.flatten() # Flatten the axes array for easier iteration

    # Iterate over each replicate and plot on the corresponding subplot
    for i in range(num_replicates):
        # Get the data for the current replicate
        replicate_data = df.iloc[:, i]

        # Get the top n genes
        top_genes = replicate_data.nlargest(n_genes).index
        top_scores = replicate_data.nlargest(n_genes).values

        # Sort the top scores in ascending order
        sorted_indices = top_scores.argsort()
        top_genes = top_genes[sorted_indices]
        top_scores = top_scores[sorted_indices]

        # Plot the gene names and scores
        ax = axes[i]
        for j in range(len(top_genes)):
            ax.text(top_scores[j], j, top_genes[j], ha='right', va='center')

        usage_key = f"usage_{i+1}"
        title = rename_dict.get(usage_key, usage_key) if rename_dict else usage_key
        ax.set_title(title)
        ax.set_ylim(-1, len(top_genes))

        x_min_percentage = 0.28
        x_min = min(top_scores) - (max(top_scores) - min(top_scores)) * x_min_percentage
        ax.set_xlim(x_min, max(top_scores) + 0.00003)
        
        ax.tick_params(axis='x', rotation=45)
        ax.set_yticks([])
        ax.set_yticklabels([])

    # For any remaining subplots (if num_replicates < fig_rows * fig_cols), make them blank
    for i in range(num_replicates, len(axes)):
        axes[i].axis('off')


    # Adjust the spacing between subplots
    plt.tight_layout()
    fig.set_facecolor('white')

    output_filename = output_prefix + '_spectra_score_plots.png'
    if save_plot:
        plt.savefig(output_filename, dpi=dpi)
    plt.close(fig)