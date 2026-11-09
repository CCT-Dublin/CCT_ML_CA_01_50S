import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ******************************************************************************************************************** #
#                                              System Logging                                                          #
# ******************************************************************************************************************** #
logging.basicConfig(
    format=("%(asctime)s | %(levelname)s | File_name ~> %(module)s.py "
            "| Function ~> %(funcName)s | Line ~~> %(lineno)d  ~~>  %(message)s"),
    level=logging.INFO
)


# ******************************************************************************************************************** #
#                                               Main function                                                          #
# ******************************************************************************************************************** #
class DpGraphicBoxPlot:
    def __init__(self):
        pass


    def F_boxplot(self, data: pd.DataFrame, column_list: list) -> None:

        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        ncols = 3
        nrows = int(np.ceil(len(column_list) / ncols))

        fig, axes   = plt.subplots(
            nrows   = nrows,
            ncols   = ncols,
            figsize = (16, 5 * nrows)
        )

        axes = np.array(axes).flatten()

        for i, (ax, feature) in enumerate(zip(axes, column_list)):

            sns.boxplot(
                data    = data,
                x       = "Target",
                y       = feature,
                hue     = "Target",
                palette = {
                                0: "#6B84B9",
                                1: "#E15759",
                                2: "#83B692"
                            },
                legend = False,
                width       = 0.6,
                linewidth   = 1.2,
                showfliers  = True,
                medianprops = {
                                "color": "#333333",
                                "linewidth": 1.5
                            },
                ax          = ax
            )

            ax.set_title(
                label       = f"Distribution of {feature}",
                fontsize    = 12,
                fontweight  = "bold",
                pad         = 12
            )

            ax.set_xlabel("Student Academic Outcome", fontsize = 10, labelpad = 8)
            ax.set_ylabel("Feature Values", fontsize = 10, labelpad = 8)

            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(
                ["Dropout", "Enrolled", "Graduate"],
                fontsize = 9
            )

            ax.tick_params(axis = "y", labelsize = 9)

            ax.grid(
                axis        = "y",
                linestyle   = "--",
                alpha       = 0.3
            )

            ax.set_axisbelow(True)

            sns.despine(ax = ax)

        for ax in axes[len(column_list):]:
            ax.set_visible(False)

        fig.suptitle(
            t           = "Numerical Feature Distributions by Student Academic Outcome",
            fontsize    = 17,
            fontweight  = "bold",
            y           = 1.01
        )

        plt.tight_layout()
        plt.show()



    def F_boxplot_v1(self, data: pd.DataFrame, column_list: list, ncols: int = 3) -> None:

        nrows = int(np.ceil(len(column_list) / ncols))

        fig, axes   = plt.subplots(
            nrows   = nrows,
            ncols   = ncols,
            figsize = (16, 4 * nrows)
        )

        axes = np.array(axes).flatten()

        for i, (ax, column) in enumerate(zip(axes, column_list)):

            sns.boxplot(
                data        = data,
                x           = column,
                color       = "steelblue",
                width       = 0.5,
                linewidth   = 1.2,
                showfliers  = True,
                medianprops     = {
                                    "color": "#E15759",
                                    "linewidth": 2
                                },
                ax              = ax
            )

            ax.set_title(
                label       = f"Outlier Analysis - {column}",
                fontsize    = 12,
                fontweight  = "bold",
                pad         = 10
            )

            ax.set_xlabel(column, fontsize = 10, labelpad = 8)
            ax.set_ylabel("Values", fontsize = 10)

            ax.tick_params(axis = "both", labelsize = 9)

            ax.grid(
                axis        = "x",
                linestyle   = "--",
                alpha       = 0.35
            )

            ax.set_axisbelow(True)

            sns.despine(ax = ax)

        for ax in axes[len(column_list):]:
            ax.set_visible(False)

        fig.suptitle(
            t           = "Numerical Features - Outlier Distribution Analysis",
            fontsize    = 17,
            fontweight  = "bold",
            y           = 1.01
        )

        plt.tight_layout()
        plt.show()
