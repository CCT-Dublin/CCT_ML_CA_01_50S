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
class DpGraphicHist:
    def __init__(self):
        pass


    def F_histplot(self, data: pd.DataFrame, columns: list = None) -> None:

        if not columns:
            logging.warning("The 'columns' parameter cannot be empty.")
            return

        ncols = 3
        nrows = (len(columns) + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows   = nrows,
            ncols   = ncols,
            figsize = (16, 4 * nrows)
        )

        axes = np.array(axes).flatten()

        for i, feature in enumerate(columns):

            sns.histplot(
                data        = data,
                x           = feature,
                bins        = 50,
                kde         = True,
                stat        = 'density',
                color       = 'steelblue',
                edgecolor   = 'white',
                alpha       = 0.85,
                ax          = axes[i]
            )

            axes[i].set_title(
                label       = f'Distribution of {feature}',
                fontsize    = 12,
                fontweight  = 'bold',
                pad         = 12
            )

            axes[i].set_xlabel(feature, fontsize=10)
            axes[i].set_ylabel('Density', fontsize=10)

            axes[i].grid(
                axis        = 'y',
                linestyle   = '--',
                alpha       = 0.3
            )

            axes[i].set_axisbelow(True)

            sns.despine(ax=axes[i])

        for j in range(len(columns), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(
            t           = 'Numerical Feature Distributions',
            fontsize    = 16,
            fontweight  = 'bold',
            y           = 1.02
        )

        plt.tight_layout()
        plt.show()