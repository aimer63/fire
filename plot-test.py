# plot-test.py

import matplotlib.pyplot as plt
import numpy as np
import time # Import time module

def create_four_fantasy_plots():
    """
    Creates a single figure with four subplots displaying fantasy data,
    and shows it in an interactive window.
    """

    print("Creating four fantasy plots in one interactive window...")

    # Create a figure and a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    fig.suptitle('Fantasy Data - Four Subplots in One Window', fontsize=16)

    # --- Plot 1: Simple Line Plot ---
    x_data_1 = np.linspace(0, 10, 100)
    y_data_1 = np.sin(x_data_1)
    axes[0, 0].plot(x_data_1, y_data_1, color='blue')
    axes[0, 0].set_title('Sine Wave')
    axes[0, 0].set_xlabel('X-axis')
    axes[0, 0].set_ylabel('Y-axis')
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)

    # --- Plot 2: Scatter Plot ---
    np.random.seed(42) # for reproducibility
    x_data_2 = np.random.rand(50) * 10
    y_data_2 = np.random.rand(50) * 10 + 5
    axes[0, 1].scatter(x_data_2, y_data_2, color='red', alpha=0.7)
    axes[0, 1].set_title('Random Scatter')
    axes[0, 1].set_xlabel('Feature A')
    axes[0, 1].set_ylabel('Feature B')
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)

    # --- Plot 3: Bar Chart ---
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 12, 39]
    axes[1, 0].bar(categories, values, color='green')
    axes[1, 0].set_title('Category Values')
    axes[1, 0].set_xlabel('Category')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot 4: Histogram ---
    data_4 = np.random.normal(loc=0, scale=1, size=1000) # mean 0, std dev 1
    axes[1, 1].hist(data_4, bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Normal Distribution Histogram')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust rect to make space for suptitle

    # Display the plot in an interactive window
    plt.show(block=True) # Ensure it blocks execution until the window is closed
    # plt.pause(0.1) # This is less commonly needed if block=True is used, but can help in some edge cases.

    print("Interactive window displayed. Close the window to continue program execution.")

if __name__ == "__main__":
    create_four_fantasy_plots()