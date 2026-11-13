import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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
class DpGraphicHelpful:
    def __init__(self):
        pass

    def F_correlation_matrix(self, data:pd.DataFrame, column_list:list, _type:int = 1) -> sns.heatmap:
        match _type:
            case 1:
                correlation_matrix = data[column_list].corr(method = "pearson")
                plt.figure(figsize = (22, 18))
            case 2:
                correlation_matrix = data.select_dtypes(include=['float64', 'int64']).corr()
                plt.figure(figsize = (22, 18))
            case _:
                return

        sns.heatmap(correlation_matrix, annot = True, fmt = '.2f', cmap = 'coolwarm', square = True, cbar_kws = {"shrink": .8})
        plt.title('Heatmap of Correlation Matrix for Breast Cancer Wisconsin')
        plt.show()

    def F_plot_pca(self, cumulative_data:np.ndarray, n_components:int) -> None:
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(
            range(1, len(cumulative_data) + 1),
            cumulative_data * 100,
            marker      = "o",
            linewidth   = 2,
            label       = "Cumulative Explained Variance"
        )

        ax.axhline(
            y           = 99.5,
            color       = "indianred",
            linestyle   = "--",
            linewidth   = 1.5,
            label       = "99.5% Variance Threshold"
        )

        ax.axvline(
            x           = n_components,
            color       = "steelblue",
            linestyle   = "--",
            linewidth   = 1.5,
            label       = f"Selected Components: {n_components}"
        )

        ax.set(
            title  = "PCA — Cumulative Explained Variance",
            xlabel = "Number of Principal Components",
            ylabel = "Explained Variance (%)"
        )

        ax.set_ylim(0, 101)
        ax.set_xticks(range(1, len(cumulative_data) + 1))
        ax.grid(axis = "both", linestyle = "--", alpha = 0.4)

        ax.legend()
        sns.despine()
        plt.tight_layout()
        plt.show()


    def F_confusion_matrix(
            self,
            y_true: pd.Series,
            y_pred_1: np.array,
            y_pred_2: np.array = None,
            title_1: str = None,
            title_2: str = None,
            labels: list = None,
            class_names: list = None,
    ) -> None:

        matrices = []
        titles = []

        if title_1 is not None:
            cm_1 = confusion_matrix(
                y_true = y_true,
                y_pred = y_pred_1,
                labels = labels
            )

            matrices.append(cm_1)
            titles.append(title_1)

        if title_2 is not None:
            cm_2 = confusion_matrix(
                y_true = y_true,
                y_pred = y_pred_2,
                labels = labels
            )
            matrices.append(cm_2)
            titles.append(title_2)

        fig, axes = plt.subplots(
            nrows   = 1,
            ncols   = max(1, len(matrices)),
            figsize = (12, 6)
        )

        if len(matrices) == 1:
            axes = [axes]


        for ax, cm, title in zip(axes, matrices, titles):

            sns.heatmap(
                data        = cm,
                annot       = True,
                fmt         = "d",
                cmap        = "Blues",
                linewidths  = 0.5,
                cbar        = False,
                xticklabels = class_names,
                yticklabels = class_names,
                ax          = ax
            )

            ax.set_title(
                title,
                fontsize    = 14,
                fontweight  = "bold"
            )

            ax.set_xlabel("Predicted Label", fontsize = 11)
            ax.set_ylabel("Actual Label", fontsize = 11)

        fig.suptitle(
            "Confusion Matrix Comparison",
            fontsize    = 16,
            fontweight  = "bold"
        )

        plt.tight_layout()
        plt.show()



    def F_comparison_plot(self, data: pd.DataFrame) -> None:

        metrics = ["CV Mean", "Test Accuracy"]

        model_order = data["Model"].unique()
        test_size_order = ["5%", "15%", "25%"]

        sns.set_theme(style="whitegrid", context="notebook")

        # Convert evaluation scores to percentage
        df_plot = data.copy()
        df_plot[metrics] = df_plot[metrics].mul(100)

        fig, axes = plt.subplots(
            nrows   = 1,
            ncols   = 2,
            figsize = (16, 7)
        )

        for ax, metric in zip(axes, metrics):

            sns.barplot(
                data        = df_plot,
                x           = "Test Size",
                y           = metric,
                hue         = "Model",
                order       = test_size_order,
                hue_order   = model_order,
                palette     = "Set2",
                edgecolor   = "white",
                linewidth   = 0.8,
                ax          = ax
            )

            # Bar labels
            for container in ax.containers:

                ax.bar_label(
                    container   = container,
                    fmt         = "%.2f%%",
                    fontsize    = 9,
                    fontweight  = "bold",
                    padding     = 3
                )

            ax.set_title(
                label       = metric,
                fontsize    = 14,
                fontweight  = "bold",
                pad         = 12
            )

            ax.set_xlabel("Test Size", fontsize=11, labelpad=10)
            ax.set_ylabel("Evaluation Score (%)", fontsize=11, labelpad=10)

            ax.set_ylim(0, 110)

            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.0f}%")
            )

            ax.grid(
                axis        = "y",
                linestyle   = "--",
                alpha       = 0.3
            )

            ax.grid(axis="x", visible=False)

            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=9)

            sns.despine(ax=ax)

            # Remove individual legends
            ax.get_legend().remove()

        # Shared legend
        handles, labels = axes[0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            title           = "Classification Model",
            loc             = "lower center",
            bbox_to_anchor  = (0.5, 0.015),
            ncol            = 2,
            fontsize        = 10,
            title_fontsize  = 10,
            frameon         = False
        )

        fig.suptitle(
            "Classification Model Performance Comparison",
            fontsize    = 17,
            fontweight  = "bold",
            y           = 0.98
        )

        # Reserve space for title and shared legend
        fig.tight_layout(rect=[0, 0.12, 1, 0.93])

        plt.show()

