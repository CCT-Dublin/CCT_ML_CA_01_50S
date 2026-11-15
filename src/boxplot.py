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

        fig, axes = plt.subplots(
            nrows   = nrows,
            ncols   = ncols,
            figsize = (16, 4 * nrows)
        )

        axes = np.array(axes).flatten()

        for i, (ax, column) in enumerate(zip(axes, column_list)):

            min_value = data[column].min()
            max_value = data[column].max()

            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)

            lower_whisker = data.loc[data[column] >= lower_bound, column].min()
            upper_whisker = data.loc[data[column] <= upper_bound, column].max()

            sns.boxplot(
                data        = data,
                x           = column,
                color       = "steelblue",
                width       = 0.5,
                linewidth   = 1.2,
                showfliers  = True,
                medianprops = {
                                "color": "#E15759",
                                "linewidth": 2
                                },
                ax          = ax
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

            ax.text(
                x                   = 0.98,
                y                   = 0.95,
                s                   = f"Lower Whisker: {lower_whisker:,.3f}\nUpper Whisker: {upper_whisker:,.3f}\nOutlier after UW: {max_value:,.3f}",
                transform           = ax.transAxes,
                fontsize            = 9,
                fontweight          = "bold",
                verticalalignment   = "top",
                horizontalalignment = "right",
                bbox = dict(
                    boxstyle    = "round,pad=0.4",
                    facecolor   = "white",
                    edgecolor   = "gray",
                    alpha       = 0.9
                )
            )

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

    def F_boxplot_v2(self, pca_data: pd.DataFrame, pca_list: list) -> None:
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(18, 13)
        )
        
        # Consistent color palette
        palette = {
            'Legitimate': '#6B84B9',
            'Fraudulent': '#E15759'
        }
        
        for i, (ax, features) in enumerate(zip(
            axes,
            [pca_list[:14], pca_list[14:]]
        )):
        
            plot_data = pca_data.melt(
                id_vars='Diagnosis',
                value_vars=features,
                var_name='Feature',
                value_name='Value'
            )
        
            plot_data['Transaction Type'] = (
                plot_data['Diagnosis'].map({
                    0: 'Legitimate',
                    1: 'Fraudulent'
                })
            )
        
            sns.boxplot(
                data=plot_data,
                x='Feature',
                y='Value',
                hue='Transaction Type',
                palette=palette,
                width=0.7,
                linewidth=1.2,
                showfliers=False,
                medianprops={
                    'color': '#333333',
                    'linewidth': 1.5
                },
                ax=ax
            )
        
            # Individual titles
            ax.set_title(
                f'PCA Components {i * 14 + 1}–{i * 14 + len(features)}',
                fontsize=13,
                fontweight='bold',
                pad=12
            )
        
            ax.set_xlabel('PCA Components', fontsize=11, labelpad=8)
            ax.set_ylabel('Component Values', fontsize=11, labelpad=8)
        
            ax.tick_params(
                axis='x',
                labelsize=9,
                rotation=0
            )
        
            ax.tick_params(axis='y', labelsize=10)
        
            ax.grid(
                axis='y',
                linestyle='--',
                alpha=0.25
            )
        
            ax.set_axisbelow(True)
        
            # Keep only one legend
            if i == 0:
                ax.legend(
                    title='Transaction Diagnosis',
                    frameon=False,
                    loc='upper right',
                    fontsize=10,
                    title_fontsize=10
                )
            else:
                ax.get_legend().remove()
        
            sns.despine(ax=ax)
        
        fig.suptitle(
            'PCA Component Distributions by Transaction Diagnosis',
            fontsize=17,
            fontweight='bold',
            y=0.99
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()