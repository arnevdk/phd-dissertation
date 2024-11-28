import matplotlib.pyplot as plt
# import tikzplotlib
from cycler import cycler

mm = 1 / 25.4  # centimeters in inches
pagesize = (160 * mm, 240 * mm)
# colors
colors = dict(
    darkgray="#8D807A",
    lightgray="#E0E0E0",
    accent1="#F29018",
    accent2="#F63D1E",
    accent3="#78479D",
    accent4="#74A4BC",
)
diverging_cmap = "RdYlBu_r"


# Function to configure matplotlib styles
def configure_matplotlib_style():
    custom_color_palette = [
        colors[c] for c in ["accent1", "accent2", "accent3", "accent4"]
    ]

    # Configure global settings
    font_size = 9

    plt.rcParams.update(
        {
            "text.usetex": True,  # use LaTeX to write all text
            "pgf.texsystem": "pdflatex",
            # "axes.grid": False,  # Disable grid lines
            "pgf.rcfonts": False,
            "axes.titlelocation": "left",  # Axis title alignment left
            "figure.facecolor": "white",  # Background color for the figure
            "image.cmap": "inferno",  # Set default colormap to Plasma
            "axes.prop_cycle": cycler(
                color=custom_color_palette
            ),  # Custom color palette
            "figure.titlesize": 11,  # Figure title font size
            "figure.titleweight": "bold",  # Figure title bold
            "xtick.direction": "in",  # X-axis ticks inward
            "ytick.direction": "in",  # Y-axis ticks inward
            "xtick.major.size": 3,  # Major tick size for x-axis
            "ytick.major.size": 3,  # Major tick size for y-axis
            "text.latex.preamble": """
                \\usepackage[scale=.8]{opensans}
                \\renewcommand{\sffamily}{\opensans}
                \\usepackage{sansmath}
                \\sansmath""",
            "axes.edgecolor": colors["darkgray"],  # Spine color
            "axes.labelcolor": colors["darkgray"],  # Axis label color
            "axes.titlesize": font_size,  # Axis title font size
            # "axes.titleweight": "bold",  # Axis title bold
            "axes.titlecolor": colors["darkgray"],  # Axis title color
            "axes.labelsize": font_size,  # Axis label font size
            "xtick.color": colors["darkgray"],  # X-axis tick color
            "ytick.color": colors["darkgray"],  # Y-axis tick color
            "xtick.labelsize": font_size,  # X-axis tick label size
            "ytick.labelsize": font_size,  # Y-axis tick label size
            "lines.linewidth": 1.5,  # Default line width
            "lines.markersize": 3,  # Default marker size
            "lines.markeredgewidth": 0,  # Default marker size
            "legend.fontsize": font_size,  # Legend font size
            "text.color": colors["darkgray"],  # Global text color
            "legend.frameon": False,
            # "figure.titlelocation": "left",  # Figure title alignment left
            # "figure.titlecolor": "#262321",  # Figure title color
        }
    )


# Golden ratio to set aesthetic figure height
inches_per_pt = 1 / 72.27
# textwidth_pt = 345 # manuscript
textwidth_pt = 318.66946  # booklet

textwidth_in = textwidth_pt * inches_per_pt


def save_pgf_trim(fig, ax, path, width=textwidth_in, height=None, rows=1, columns=1):
    if height is None:
        golden_ratio = (5**0.5 - 1) / 2
        height = rows * width * golden_ratio / columns
    fig.set_constrained_layout(True)
    print(width)
    print(height)
    fig.set_size_inches(width, height)
    plt.draw()
    dpi = fig.dpi
    margin_left = ax.get_window_extent().x0 / dpi
    print(margin_left)
    fig.set_size_inches(width + margin_left, height)
    fig.patch.set_alpha(0)
    plt.savefig(path, format="pgf", bbox_inches="tight", pad_inches=0)
    prepend = "\\hspace{-" + str(margin_left) + "in}\n%"
    with open(path, "r") as original:
        data = original.read()
    with open(path, "w") as modified:
        modified.write(prepend + data)


def remove_tickspec(input_string):
    # List of lines to remove
    lines_to_remove = [
        "xtick style={color=black},",
        "ytick style={color=black},",
        "tick align=outside,",
        "tick pos=left,",
    ]

    # Split the input string by lines
    lines = input_string.splitlines()

    # Filter out the lines that match any in the list to remove
    filtered_lines = [line for line in lines if line.strip() not in lines_to_remove]

    # Join the filtered lines back into a single string
    output_string = "\n".join(filtered_lines)

    return output_string


# def save_tikz(path):
#    tikz_code = tikzplotlib.get_tikz_code(strict=False,
#                                          extra_tikzpicture_parameters=[
#'trim axis group left','trim axis group right'
#                                              ])
#    tikz_code = remove_tickspec(tikz_code)
#    with open(path, 'w') as file:
#        file.write(tikz_code)
