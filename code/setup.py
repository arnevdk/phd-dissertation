import matplotlib.pyplot as plt
from cycler import cycler

mm = 1 / 25.4  # centimeters in inches
pagesize = (160 * mm, 240 * mm)
# colors
darkgray = "#8D807A"

savefig_kws = dict(transparent=True, bbox_inches="tight", pad_inches=0)


# Function to configure matplotlib styles
def configure_matplotlib_style():
    custom_color_palette = ["#F29018", "#F63D1E", "#78479D"]  # Custom colors

    # Configure global settings
    font_size = 9

    plt.rcParams.update(
        {
            "text.usetex": True,  # use LaTeX to write all text
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            #'figure.figsize': (8, 6),                 # Plot size
            "axes.grid": False,  # Disable grid lines
            "axes.edgecolor": "#8D807A",  # Spine color
            "axes.labelcolor": "#8D807A",  # Axis label color
            "axes.titlesize": font_size,  # Axis title font size
            # "axes.titleweight": "bold",  # Axis title bold
            "axes.titlelocation": "left",  # Axis title alignment left
            "axes.titlecolor": "#8D807A",  # Axis title color
            "axes.labelsize": font_size,  # Axis label font size
            "xtick.color": "#8D807A",  # X-axis tick color
            "ytick.color": "#8D807A",  # Y-axis tick color
            "xtick.labelsize": font_size,  # X-axis tick label size
            "ytick.labelsize": font_size,  # Y-axis tick label size
            "xtick.direction": "in",  # X-axis ticks inward
            "ytick.direction": "in",  # Y-axis ticks inward
            "xtick.major.size": 6,  # Major tick size for x-axis
            "ytick.major.size": 6,  # Major tick size for y-axis
            "lines.linewidth": 1.5,  # Default line width
            "lines.markersize": 8,  # Default marker size
            "legend.fontsize": font_size,  # Legend font size
            # "legend.loc": "upper right",  # Legend position
            "legend.frameon": False,  # No frame around legend
            "axes.prop_cycle": cycler(
                color=custom_color_palette
            ),  # Custom color palette
            "text.color": "#8D807A",  # Global text color
            "figure.facecolor": "white",  # Background color for the figure
            "figure.titlesize": 11,  # Figure title font size
            "figure.titleweight": "bold",  # Figure title bold
            # "figure.titlelocation": "left",  # Figure title alignment left
            # "figure.titlecolor": "#262321",  # Figure title color
        }
    )


textwidth = 469.0
linewidth = 229.5

# Golden ratio to set aesthetic figure height
golden_ratio = (5**0.5 - 1) / 2


def set_size(width_pt=textwidth, fraction=1, aspect=golden_ratio, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * aspect * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
