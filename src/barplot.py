import logging
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
class DpGraphicBar:
    def __init__(self):
        pass

    def F_datatypes_barplot(self, data:pd.DataFrame) -> None:
        type_counts = data.dtypes.astype(str).value_counts()
        type_percentages = type_counts / type_counts.sum() * 100

        sns.set_theme(style="whitegrid", context="notebook")

        fig, ax = plt.subplots(figsize=(14, 6))

        sns.barplot(
            x       = type_counts.index,
            y       = type_counts.values,
            hue     = type_counts.index,
            palette = "Blues_r",
            legend  = False,
            ax      = ax
        )

        ax.text(
            x                   = 0.85,
            y                   = 0.95,
            s                   = f"Total Data Types: {type_counts.sum():,}",
            transform           = ax.transAxes,
            fontsize            = 12,
            fontweight          = "bold",
            verticalalignment   = "top",
            horizontalalignment = "left",
            bbox = dict(
                boxstyle    = "round,pad=0.5",
                facecolor   = "white",
                edgecolor   = "gray",
                alpha       = 0.9
            )
        )

        ax.set_title(
            label       = "Distribution of Dataset Data Types",
            fontsize    = 16,
            fontweight  = "bold",
            pad         = 15
        )

        ax.set_xlabel("Data Type", fontsize = 12, labelpad = 10)
        ax.set_ylabel("Number of Attributes", fontsize = 12, labelpad = 10)

        for i, (count, percentage) in enumerate(
            zip(type_counts.values, type_percentages.values)
        ):
            ax.annotate(
                text        = f"{count:,.0f}\n({percentage:.2f}%)",
                xy          = (i, count),
                xytext      = (0, 6),
                textcoords  = "offset points",
                ha          = "center",
                fontsize    = 10,
                fontweight  = "bold"
            )

        ax.grid(axis = "y", linestyle = "--", alpha = 0.4)
        ax.grid(axis = "x", visible = False)

        sns.despine()

        plt.tight_layout()
        plt.show()


    def F_missing_barplot(self, data:pd.DataFrame) -> None:
        missing_counts = data.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]

        if missing_counts.empty:
            logging.warning("No missing values found.")
            return

        total_missing = missing_counts.sum()

        missing_percentages = missing_counts / len(data) * 100

        sns.set_theme(style="whitegrid", context="notebook")

        fig, ax = plt.subplots(figsize=(18, 6))

        sns.barplot(
            x       = missing_counts.index,
            y       = missing_counts.values,
            hue     = missing_counts.index,
            palette = "Reds_r",
            legend  = False,
            ax      = ax
        )

        ax.set_title(
            label       = "Distribution of Missing Values",
            fontsize    = 16,
            fontweight  = "bold",
            pad         = 15
        )

        ax.set_xlabel("Attributes", fontsize = 12, labelpad = 10)
        ax.set_ylabel("Number of Missing Values", fontsize = 12, labelpad = 10)

        ax.tick_params(axis="x", rotation=30)

        ax.text(
            x                   = 0.02,
            y                   = 0.95,
            s                   = f"Total Missing Values: {total_missing:,}",
            transform           = ax.transAxes,
            fontsize            = 12,
            fontweight          = "bold",
            verticalalignment   = "top",
            horizontalalignment = "left",
            bbox = dict(
                boxstyle    = "round,pad=0.5",
                facecolor   = "white",
                edgecolor   = "gray",
                alpha       = 0.9
            )
        )

        for i, (count, percentage) in enumerate(
            zip(missing_counts.values, missing_percentages.values)
        ):
            ax.annotate(
                text        = f"{count:,.0f}\n({percentage:.2f}%)",
                xy          = (i, count),
                xytext      = (0, 6),
                textcoords  = "offset points",
                ha          = "center",
                fontsize    = 10,
                fontweight  = "bold"
            )

        ax.grid(axis = "y", linestyle = "--", alpha = 0.4)
        ax.grid(axis = "x", visible = False)

        sns.despine()

        plt.tight_layout()
        plt.show()


    def F_target_barplot(self, data:pd.DataFrame, column:str) -> None:
        label_counts = data[column].value_counts(dropna=False)
        label_percentages = label_counts / label_counts.sum() * 100

        sns.set_theme(style="whitegrid", context="notebook")

        fig, ax = plt.subplots(figsize=(14, 6))

        sns.barplot(
            x       = label_counts.index.astype(str),
            y       = label_counts.values,
            hue     = label_counts.index.astype(str),
            palette = "Blues_r",
            legend  = False,
            ax      = ax
        )

        ax.set_title(
            label       = "Distribution of cancer diagnoses",
            fontsize    = 16,
            fontweight  = "bold",
            pad         = 15
        )

        ax.text(
            x                   = 0.85,
            y                   = 0.95,
            s                   = f"Total Label Cancer Diagnoses: {label_counts.sum():,}",
            transform           = ax.transAxes,
            fontsize            = 12,
            fontweight          = "bold",
            verticalalignment   = "top",
            horizontalalignment = "left",
            bbox = dict(
                boxstyle    = "round,pad=0.5",
                facecolor   = "white",
                edgecolor   = "gray",
                alpha       = 0.9
            )
        )

        ax.set_xlabel("Cancer Diagnoses Outcome", fontsize = 12, labelpad = 10)
        ax.set_ylabel("Number of Diagnoses", fontsize = 12, labelpad = 10)

        for i, (count, percentage) in enumerate(
            zip(label_counts.values, label_percentages.values)
        ):
            ax.annotate(
                text        = f"{count:,.0f}\n({percentage:.2f}%)",
                xy          = (i, count),
                xytext      = (0, 6),
                textcoords  = "offset points",
                ha          = "center",
                fontsize    = 10,
                fontweight  = "bold"
            )

        ax.grid(axis = "y", linestyle = "--", alpha = 0.4)
        ax.grid(axis = "x", visible = False)

        sns.despine()

        plt.tight_layout()
        plt.show()