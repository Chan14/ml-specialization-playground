from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Define the path to the dataset
# This constructs the path relative to the current script's location
file_path = Path(__file__).resolve().parent / "data" / "ames_housing_subset.csv"

try:
    # === 1. Load the dataset ===
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully.")

    # === 2. Set feature list and target ===
    features = [
        "Lot Area",
        "Gr Liv Area",
        "Overall Qual",
        "Total Bsmt SF",
        "Year Built",
        "Garage Cars",
    ]
    target = "SalePrice"

    # === 3. Set up the subplot grid ===
    fig, axes = plt.subplots(4, 3, figsize=(17, 9))
    fig.suptitle(
        "Ames Housing: Feature Distributions & Relationships",
        fontsize=16,
        fontweight="bold",
    )
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # === 4. Plot Histograms (First two rows: axes 0 to 5) ===
    for i, feature in enumerate(features):
        ax = axes[i]
        ax.hist(df[feature], bins=20, color="skyblue", edgecolor="black")
        ax.set_title(f"Histogram of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")

    # === 5. Plot Scatterplots vs SalePrice (Next two rows: rows 2–3 or axes 6 to 11) ===
    print("Generating scatter plots...")
    # Initialize the first scatter plot's y-axis for sharing
    # We'll use the 7th subplot (index 6) for the first scatter plot
    first_scatter_ax = axes[len(features)]  # This will be axes[6]

    for i, feature in enumerate(features):
        ax = axes[len(features) + i]  # Start from axes[6]
        # Share y-axis with the first scatter plot axis to ensure SalePrice scale is consistent
        if i > 0:
            ax.sharey(first_scatter_ax)
        ax.scatter(df[feature], df[target], alpha=0.7, color="lightcoral")
        ax.set_title(f"Scatter plot: {feature} vs {target}")
        ax.set_xlabel(feature)
        ax.set_ylabel(target)  # Only set label for shared y-axis or first plot

    # Adjust subplot parameters for a tighter layout, making space for the suptitle
    plt.subplots_adjust(top=0.98)  # Adjust top to create space for suptitle
    # Adjust rect to ensure suptitle is not cut off by tight_layout
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Define the output image file name
    output_image_filename = (
        Path(__file__).resolve().parent / "data" / "ames_housing_feature_plots.png"
    )

    # Save the figure
    plt.savefig(output_image_filename)
    print(f"\nVisualizations saved to '{output_image_filename}'")

    # Show the plot (optional, remove if only need to save)
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print(
        "Please ensure the 'data' folder exists in the current directory and 'ames_housing_subset.csv' is inside it."
    )
except KeyError as e:
    print(f"Error: A column was not found. Please check your 'features' list.")
    print(f"Missing column: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
