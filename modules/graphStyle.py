import matplotlib.pyplot as plt
import cmasher as cmr 
import numpy as np

def set_plot_style(num_colors: int=4, cmap: str='cmr.bubblegum_r', cmap_range: tuple[float, float]=(0.1, 0.9)):
    # Generate a colormap within the given range
    cm = cmr.get_sub_cmap(cmap, *cmap_range)
    colors = [cm(i / (num_colors - 1)) for i in range(num_colors)]  
    # Apply global Matplotlib settings
    plt.rcParams.update({
        'axes.prop_cycle': plt.cycler(color=colors),
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'image.cmap': cmap
    })
    plt.rcParams['pgf.texsystem'] = 'pdflatex'
    plt.rcParams.update({'font.family': 'serif', 'font.size': 18,
        'axes.labelsize': 20,'axes.titlesize': 24, 'figure.titlesize' : 28})
    plt.rcParams['text.usetex'] = True

def add_subplot_labels(axes, x_pos=-0.3, y_pos=1.05, fontsize=20):
    """
    Add labels (a, b, c, etc.) to the upper left corner of each subplot.
    
    Parameters:
    -----------
    axes : array-like
        The matplotlib axes objects to add labels to.
    x_pos : float, optional
        The x position of the label in axis coordinates (0-1). Default is 0.05.
    y_pos : float, optional
        The y position of the label in axis coordinates (0-1). Default is 0.9.
    fontsize : int, optional
        The font size of the labels. Default is 12.
    """
    # Convert to array if it's a single axis
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    
    # Flatten in case it's a 2D array of axes
    axes_flat = axes.flatten()
    
    # Add labels to each subplot
    for i, ax in enumerate(axes_flat):
        label = chr(97 + i) + ")"  # 97 is ASCII code for 'a'
        ax.text(x_pos, y_pos, label, transform=ax.transAxes, 
                fontsize=fontsize, fontweight='bold')
        
def add_colorbar_to_figure(fig, contour_plot, label=r'$|\rho|$', vmin=None, vmax=None):
    """
    Add a single colorbar to the right side of a figure containing contour plots.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to add the colorbar to
    contour_plot : matplotlib.contourf
        Any of the contour plots (will use its colormap)
    label : str
        Label for the colorbar
    vmin, vmax : float, optional
        Manually set colorbar limits
    """
    # Adjust the layout to make room for the colorbar
    fig.subplots_adjust(right=0.85)
    
    # Create axes for the colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    
    # Add colorbar with optional manual limits
    if vmin is not None and vmax is not None:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap='cmr.bubblegum')
        cbar = fig.colorbar(mappable, cax=cbar_ax)
    else:
        cbar = fig.colorbar(contour_plot, cax=cbar_ax)
    
    # Add label to colorbar
    cbar.set_label(label, rotation=0, labelpad=15)
    
    return cbar
