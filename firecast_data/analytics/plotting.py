#
# Copyright (c) 2025-Present aimer <63aimer@gmail.com
# All rights reserved.
#
# Licensed under GNU Affero General Public License v3 (AGPLv3).
#
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# This local import is assumed to be available, similar to data_metrics.py
from firecast.utils.colors import get_color


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame, interactive: bool
) -> None:
    """
    Generates and saves a heatmap of the asset correlation matrix.

    Args:
        correlation_matrix: The correlation matrix to plot.
        interactive: If True, show the plot window.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = get_color("mocha", "crust")
    plt.rcParams["axes.facecolor"] = get_color("mocha", "crust")

    gradient = LinearSegmentedColormap.from_list(
        "gradient",
        [
            get_color("mocha", "red"),
            get_color("mocha", "text"),
            get_color("latte", "mauve"),
        ],
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        vmin=-1,
        vmax=1,
        cmap=gradient,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
    )
    plt.title("Asset Correlation Matrix")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(filepath)
    print(f"\nCorrelation heatmap saved to '{filepath}'")
    if interactive:
        plt.show()
    plt.close()


def plot_asset_prices(price_df: pd.DataFrame, interactive: bool) -> None:
    """
    Generates and saves a plot of prices for each asset to help spot anomalies.

    Args:
        price_df: DataFrame of asset prices.
        interactive: If True, show the plot windows.
    """
    output_dir = "output/price_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving price plots to '{output_dir}/' for inspection.")

    for asset in price_df.columns:
        asset_prices = price_df[asset].dropna()

        if asset_prices.empty:
            continue

        plt.style.use("dark_background")
        plt.rcParams["figure.facecolor"] = get_color("mocha", "crust")
        plt.rcParams["axes.facecolor"] = get_color("mocha", "crust")

        plt.figure(figsize=(15, 7))
        plt.plot(
            asset_prices,
            color=get_color("mocha", "blue"),
            linewidth=0.8,
        )

        plt.title(f"Price History for {asset}", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        safe_asset_name = asset.replace("/", "_").replace(" ", "_")
        filepath = os.path.join(output_dir, f"price_history_{safe_asset_name}.png")
        plt.savefig(filepath)
        if interactive:
            plt.show()
        plt.close()


def plot_asset_return_distributions(
    window_returns_df: pd.DataFrame, interactive: bool
) -> None:
    """
    Plots the kernel density estimate of the return distribution for each asset.
    """
    output_dir = "output/asset_return_distributions"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving asset return distribution plots to '{output_dir}/'")

    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = get_color("mocha", "crust")
    plt.rcParams["axes.facecolor"] = get_color("mocha", "crust")

    for asset in window_returns_df.columns:
        asset_returns = window_returns_df[asset].dropna()

        if asset_returns.empty:
            continue

        plt.figure(figsize=(10, 6))

        sns.kdeplot(
            x=asset_returns, color=get_color("mocha", "blue"), fill=True, alpha=0.3
        )

        mean_return = asset_returns.mean()
        var_95 = asset_returns.quantile(0.05)

        plt.axvline(
            mean_return,
            color=get_color("mocha", "green"),
            linestyle="--",
            label=f"Mean: {mean_return:.2%}",
        )
        plt.axvline(
            var_95,
            color=get_color("mocha", "red"),
            linestyle="--",
            label=f"VaR 95%: {var_95:.2%}",
        )
        plt.axvline(0, color=get_color("mocha", "text"), linestyle=":", alpha=0.5)

        plt.title(f"Return Distribution for {asset}")
        plt.xlabel("Annualized Return")
        plt.ylabel("Density")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        safe_asset_name = asset.replace("/", "_").replace(" ", "_")
        filepath = os.path.join(output_dir, f"return_dist_{safe_asset_name}.png")
        plt.savefig(filepath)

        if interactive:
            plt.show()

        plt.close()


def plot_assets_boxplot(window_returns_df: pd.DataFrame, interactive: bool) -> None:
    """
    Plots a boxplot of the return distributions for all assets.
    Only shows the plot interactively if requested.
    """
    if not interactive:
        return

    plt.style.use("dark_background")
    plt.figure(figsize=(12, 8))

    data = [window_returns_df[col].dropna() for col in window_returns_df.columns]
    labels = list(window_returns_df.columns)
    box_colors = [get_color("mocha", "blue") for _ in labels]

    box = plt.boxplot(data, patch_artist=True, widths=0.2)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=90)
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    plt.title("Return Distributions of All Assets (Boxplot)")
    plt.ylabel("Annualized Return")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if interactive:
        plt.show()

    plt.close()


def plot_efficient_frontier(
    portfolios_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    min_vol_portfolio: pd.Series,
    max_sharpe_portfolio: pd.Series,
    max_var_portfolio: pd.Series,
    max_cvar_portfolio: pd.Series,
    max_adj_sharpe_portfolio: pd.Series,
) -> None:
    """
    Plots the efficient frontier from simulated portfolios and individual assets.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    plt.figure(figsize=(12, 8))

    # Plot the simulated portfolios
    plt.scatter(
        portfolios_df["Volatility"],
        portfolios_df["Return"],
        c=portfolios_df["Sharpe"],
        cmap="viridis",
        marker=".",
        alpha=0.7,
    )

    # Plot the individual assets
    plt.scatter(
        summary_df["Rolling Volatility"],
        summary_df["Rolling Return"],
        marker="X",
        color="red",
        s=100,
        label="Individual Assets",
    )

    # Label the individual assets
    for asset, row in summary_df.iterrows():
        plt.text(
            row["Rolling Volatility"] * 1.01,
            row["Rolling Return"],
            str(asset),
            fontsize=9,
            color=get_color("mocha", "text"),
        )

    # Highlight the minimum volatility portfolio
    plt.scatter(
        min_vol_portfolio["Volatility"],
        min_vol_portfolio["Return"],
        marker="*",
        color=get_color("mocha", "green"),
        s=250,
        label="Minimum Volatility",
        zorder=5,
    )

    # Highlight the maximum Sharpe ratio portfolio
    plt.scatter(
        max_sharpe_portfolio["Volatility"],
        max_sharpe_portfolio["Return"],
        marker="*",
        color=get_color("mocha", "yellow"),
        s=250,
        label="Maximum Sharpe Ratio",
        zorder=5,
    )

    # Highlight the maximum VaR 95% portfolio
    plt.scatter(
        max_var_portfolio["Volatility"],
        max_var_portfolio["Return"],
        marker="*",
        color=get_color("mocha", "mauve"),
        s=250,
        label="Maximum VaR 95%",
        zorder=5,
    )

    # Highlight the maximum CVaR 95% portfolio
    plt.scatter(
        max_cvar_portfolio["Volatility"],
        max_cvar_portfolio["Return"],
        marker="*",
        color=get_color("mocha", "blue"),
        s=250,
        label="Maximum CVaR 95%",
        zorder=5,
    )

    # Highlight the maximum Adjusted Sharpe portfolio
    plt.scatter(
        max_adj_sharpe_portfolio["Volatility"],
        max_adj_sharpe_portfolio["Return"],
        marker="*",
        color=get_color("mocha", "peach"),
        s=250,
        label="Maximum Adjusted Sharpe",
        zorder=5,
    )

    plt.title("Monte Carlo Simulation for Efficient Frontier")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Expected Annualized Return")
    plt.colorbar(label="Sharpe Ratio")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "efficient_frontier.png")
    plt.savefig(filepath)
    print(f"\nEfficient frontier plot saved to '{filepath}'")


def plot_efficient_frontier_var(
    portfolios_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    min_vol_portfolio: pd.Series,
    max_sharpe_portfolio: pd.Series,
    max_var_portfolio: pd.Series,
    max_cvar_portfolio: pd.Series,
    max_adj_sharpe_portfolio: pd.Series,
) -> None:
    """
    Plots the efficient frontier using VaR 95% on the x-axis.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    plt.figure(figsize=(12, 8))

    # Plot the simulated portfolios
    plt.scatter(
        portfolios_df["VaR 95%"],
        portfolios_df["Return"],
        c=portfolios_df["Sharpe"],
        cmap="viridis",
        marker=".",
        alpha=0.7,
    )

    # Plot the individual assets
    plt.scatter(
        summary_df["Rolling VaR 95%"],
        summary_df["Rolling Return"],
        marker="X",
        color="red",
        s=100,
        label="Individual Assets",
    )

    # Label the individual assets
    for asset, row in summary_df.iterrows():
        plt.text(
            row["Rolling VaR 95%"] * 1.01,
            row["Rolling Return"],
            str(asset),
            fontsize=9,
            color=get_color("mocha", "text"),
        )

    # Highlight the minimum volatility portfolio
    plt.scatter(
        min_vol_portfolio["VaR 95%"],
        min_vol_portfolio["Return"],
        marker="*",
        color=get_color("mocha", "green"),
        s=250,
        label="Minimum Volatility Portfolio",
        zorder=5,
    )

    # Highlight the maximum Sharpe ratio portfolio
    plt.scatter(
        max_sharpe_portfolio["VaR 95%"],
        max_sharpe_portfolio["Return"],
        marker="*",
        color=get_color("mocha", "yellow"),
        s=250,
        label="Maximum Sharpe Ratio Portfolio",
        zorder=5,
    )

    # Highlight the maximum VaR 95% portfolio
    plt.scatter(
        max_var_portfolio["VaR 95%"],
        max_var_portfolio["Return"],
        marker="*",
        color=get_color("mocha", "mauve"),
        s=250,
        label="Maximum VaR 95% Portfolio",
        zorder=5,
    )

    # Highlight the maximum CVaR 95% portfolio
    plt.scatter(
        max_cvar_portfolio["VaR 95%"],
        max_cvar_portfolio["Return"],
        marker="*",
        color=get_color("mocha", "blue"),
        s=250,
        label="Maximum CVaR 95% Portfolio",
        zorder=5,
    )

    # Highlight the maximum Adjusted Sharpe portfolio
    plt.scatter(
        max_adj_sharpe_portfolio["VaR 95%"],
        max_adj_sharpe_portfolio["Return"],
        marker="*",
        color=get_color("mocha", "peach"),
        s=250,
        label="Maximum Adjusted Sharpe Portfolio",
        zorder=5,
    )

    plt.title("Efficient Frontier (Return vs. VaR 95%)")
    plt.xlabel("VaR 95% (5th Percentile of Annualized Returns)")
    plt.ylabel("Expected Annualized Return")
    plt.colorbar(label="Sharpe Ratio")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "efficient_frontier_var.png")
    plt.savefig(filepath)
    print(f"\nEfficient frontier (VaR) plot saved to '{filepath}'")


def plot_portfolios_return_distributions(
    min_vol_portfolio: pd.Series,
    max_sharpe_portfolio: pd.Series,
    max_var_portfolio: pd.Series,
    max_cvar_portfolio: pd.Series,
    max_adj_sharpe_portfolio: pd.Series,
    window_returns_df: pd.DataFrame,
) -> None:
    """
    Plots the kernel density estimate of the return distributions for the optimal portfolios,
    and adds a stacked horizontal boxplot below, sharing the x-axis.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    _, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    portfolios = {
        "Minimum Volatility": (min_vol_portfolio, get_color("mocha", "green")),
        "Maximum Sharpe Ratio": (max_sharpe_portfolio, get_color("mocha", "yellow")),
        "Maximum VaR 95%": (max_var_portfolio, get_color("mocha", "mauve")),
        "Maximum CVaR 95%": (max_cvar_portfolio, get_color("mocha", "blue")),
        "Maximum Adjusted Sharpe": (
            max_adj_sharpe_portfolio,
            get_color("mocha", "peach"),
        ),
    }

    # KDE plot
    for name, (portfolio, color) in portfolios.items():
        portfolio_returns = window_returns_df.dot(portfolio["Weights"])
        sns.kdeplot(
            portfolio_returns, label=name, color=color, fill=True, alpha=0.3, ax=ax1
        )
        ax1.axvline(
            portfolio_returns.mean(),
            color=color,
            linestyle="--",
            alpha=0.8,
            linewidth=1.0,
        )

    ax1.set_title("Return Distributions of Optimal Portfolios")
    ax1.set_xlabel("Annualized Return")
    ax1.set_ylabel("Density")
    ax1.axvline(0, color=get_color("mocha", "red"), linestyle="--", alpha=0.7)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Boxplot
    data = []
    box_colors = []
    labels = []
    for name, (portfolio, color) in portfolios.items():
        portfolio_returns = window_returns_df.dot(portfolio["Weights"])
        data.append(portfolio_returns)
        box_colors.append(color)
        labels.append(name)

    box = ax2.boxplot(data, vert=False, patch_artist=True)
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax2.set_xlabel("Annualized Return")
    ax2.set_yticks(range(1, len(labels) + 1))
    ax2.set_yticklabels(labels)
    ax2.set_title("Boxplot")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "return_distributions_stacked_boxplot.png")
    plt.savefig(filepath)
    print(f"\nReturn distributions + boxplot saved to '{filepath}'")


def plot_portfolio_returns_over_time(
    min_vol_portfolio: pd.Series,
    max_sharpe_portfolio: pd.Series,
    max_var_portfolio: pd.Series,
    max_cvar_portfolio: pd.Series,
    max_adj_sharpe_portfolio: pd.Series,
    window_returns_df: pd.DataFrame,
    window_years: int,
) -> None:
    """
    Plots the historical windowed returns for the three optimal portfolios.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    plt.figure(figsize=(15, 7))

    portfolios = {
        "Minimum Volatility": (min_vol_portfolio, get_color("mocha", "green")),
        "Maximum Sharpe Ratio": (max_sharpe_portfolio, get_color("mocha", "yellow")),
        "Maximum VaR 95%": (max_var_portfolio, get_color("mocha", "mauve")),
        "Maximum CVaR 95%": (max_cvar_portfolio, get_color("mocha", "blue")),
        "Maximum Adjusted Sharpe": (
            max_adj_sharpe_portfolio,
            get_color("mocha", "peach"),
        ),
    }

    for name, (portfolio, color) in portfolios.items():
        portfolio_returns = window_returns_df.dot(portfolio["Weights"])
        plt.plot(
            portfolio_returns.index,
            portfolio_returns,
            label=name,
            color=color,
            linewidth=1.2,
        )

    plt.title("Historical Windowed Returns of Optimal Portfolios")
    plt.xlabel("Window End Date")
    plt.ylabel(f"{window_years}-Year Rolling Annualized Return")
    plt.axhline(0, color=get_color("mocha", "red"), linestyle="--", alpha=0.7)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "portfolio_returns_over_time.png")
    plt.savefig(filepath)
    print(f"\nPortfolio returns over time plot saved to '{filepath}'")


def plot_portfolios_correlation_heatmap(
    min_vol_portfolio: pd.Series,
    max_sharpe_portfolio: pd.Series,
    max_var_portfolio: pd.Series,
    max_cvar_portfolio: pd.Series,
    max_adj_sharpe_portfolio: pd.Series,
    window_returns_df: pd.DataFrame,
) -> None:
    """
    For each winning portfolio, plot a heatmap of the correlation matrix
    among the assets included in that portfolio (weights > 0).
    Always shows the plot interactively.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    portfolios = {
        "Minimum Volatility": min_vol_portfolio,
        "Maximum Sharpe Ratio": max_sharpe_portfolio,
        "Maximum VaR 95%": max_var_portfolio,
        "Maximum CVaR 95%": max_cvar_portfolio,
        "Maximum Adjusted Sharpe": max_adj_sharpe_portfolio,
    }

    for name, portfolio in portfolios.items():
        plt.style.use("dark_background")
        plt.rcParams["figure.facecolor"] = get_color("mocha", "crust")
        plt.rcParams["axes.facecolor"] = get_color("mocha", "crust")

        weights = pd.Series(portfolio["Weights"], index=window_returns_df.columns)
        selected_assets = weights[weights > 0.0001].index.tolist()
        asset_returns_df = window_returns_df[selected_assets]

        correlation_matrix = asset_returns_df.corr()

        gradient = LinearSegmentedColormap.from_list(
            "gradient",
            [
                get_color("mocha", "red"),
                get_color("mocha", "text"),
                get_color("latte", "mauve"),
            ],
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            vmin=-1,
            vmax=1,
            cmap=gradient,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": "Correlation"},
        )
        plt.title(f"Correlation Matrix: {name} Portfolio Assets")
        plt.tight_layout()

        safe_name = name.strip().replace(" ", "_").lower()
        filepath = os.path.join(
            output_dir, f"correlation_heatmap_{safe_name}_assets.png"
        )
        plt.savefig(filepath)
        print(f"\nCorrelation heatmap for {name} assets saved to '{filepath}'")


def plot_single_portfolio_return_distribution(
    portfolio: pd.Series,
    name: str,
    window_returns_df: pd.DataFrame,
) -> None:
    """
    Plots the kernel density estimate and horizontal boxplot of the return distribution
    for a manual portfolio as two stacked plots with aligned x-axis. Always shows the plot interactively.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    portfolio_returns = window_returns_df.dot(portfolio["Weights"])

    plt.style.use("dark_background")
    _, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # KDE plot
    sns.kdeplot(
        portfolio_returns,
        label=name,
        color=get_color("mocha", "blue"),
        fill=True,
        alpha=0.3,
        ax=ax1,
    )
    mean_return = portfolio_returns.mean()
    ax1.axvline(
        mean_return,
        color=get_color("mocha", "green"),
        linestyle="--",
        label=f"Mean: {mean_return:.2%}",
    )
    ax1.axvline(0, color=get_color("mocha", "red"), linestyle="--", alpha=0.7)
    ax1.set_title(f"Return Distribution for {name}")
    ax1.set_xlabel("Annualized Return")
    ax1.set_ylabel("Density")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Boxplot
    ax2.boxplot(
        portfolio_returns,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor=get_color("mocha", "blue"), alpha=0.4),
        medianprops=dict(color=get_color("mocha", "green")),
    )
    ax2.set_xlabel("Annualized Return")
    ax2.set_yticks([])
    ax2.set_title("Boxplot")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    safe_name = name.strip().replace(" ", "_").lower()
    filepath = os.path.join(output_dir, f"return_distribution_{safe_name}.png")
    plt.savefig(filepath)
    print(f"\nReturn distribution saved to '{filepath}'")


def plot_single_portfolio_returns_over_time(
    portfolio: pd.Series,
    name: str,
    window_returns_df: pd.DataFrame,
    window_years: int,
) -> None:
    """
    Plots the historical windowed returns for a single portfolio.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    plt.figure(figsize=(15, 7))

    portfolio_returns = window_returns_df.dot(portfolio["Weights"])
    plt.plot(
        portfolio_returns.index,
        portfolio_returns,
        label=name,
        color=get_color("mocha", "blue"),
        linewidth=1.2,
    )

    plt.title(f"Historical Windowed Returns of {name}")
    plt.xlabel("Window End Date")
    plt.ylabel(f"{window_years}-Year Rolling Annualized Return")
    plt.axhline(0, color=get_color("mocha", "red"), linestyle="--", alpha=0.7)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()

    safe_name = name.strip().replace(" ", "_").lower()
    filepath = os.path.join(output_dir, f"portfolio_returns_over_time_{safe_name}.png")
    plt.savefig(filepath)
    print(f"\nPortfolio returns over time plot saved to '{filepath}'")


def plot_single_portfolio_correlation_heatmap(
    portfolio: pd.Series,
    name: str,
    window_returns_df: pd.DataFrame,
) -> None:
    """
    Plots a heatmap of the correlation matrix among assets included in a manual portfolio (weights > 0).
    Always shows the plot interactively.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    weights = pd.Series(portfolio["Weights"], index=window_returns_df.columns)
    selected_assets = weights[weights > 0.0001].index.tolist()
    asset_returns_df = window_returns_df[selected_assets]

    correlation_matrix = asset_returns_df.corr()

    gradient = LinearSegmentedColormap.from_list(
        "gradient",
        [
            get_color("mocha", "red"),
            get_color("mocha", "text"),
            get_color("latte", "mauve"),
        ],
    )

    plt.style.use("dark_background")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        vmin=-1,
        vmax=1,
        cmap=gradient,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
    )
    plt.title(f"Correlation Matrix: {name} Portfolio Assets")
    plt.tight_layout()

    safe_name = name.strip().replace(" ", "_").lower()
    filepath = os.path.join(output_dir, f"correlation_heatmap_{safe_name}_assets.png")
    plt.savefig(filepath)
    print(f"\nCorrelation heatmap for {name} assets saved to '{filepath}'")


def plot_annealing_convergence(
    description: str,
    best_cost_history: List[float],
    current_cost_history: List[float],
    temp_history: List[float],
    acceptance_prob_history: List[Tuple[int, float]],
) -> None:
    """
    Plots the convergence metrics of the simulated annealing algorithm.
    """
    output_dir = "output/annealing_convergence"
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(
        f"Simulated Annealing Convergence for {description.strip()}", fontsize=16
    )

    iterations = range(len(best_cost_history))

    # Plot 1: Cost vs. Iteration
    ax1.plot(
        iterations,
        current_cost_history,
        label="Current Cost",
        color=get_color("mocha", "blue"),
        alpha=0.6,
        linewidth=0.5,
    )
    ax1.plot(
        iterations,
        best_cost_history,
        label="Best Cost",
        color=get_color("mocha", "green"),
        linewidth=1.5,
    )
    ax1.set_ylabel("Cost (Objective Value)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Plot 2: Temperature vs. Iteration
    ax2.plot(
        iterations, temp_history, label="Temperature", color=get_color("mocha", "red")
    )
    ax2.set_ylabel("Temperature")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.3)

    # Plot 3: Acceptance Probability vs. Iteration
    if acceptance_prob_history:
        prob_iters, probs = zip(*acceptance_prob_history)
        ax3.scatter(
            prob_iters,
            probs,
            label="Acceptance Probability (Worse Solution)",
            color=get_color("mocha", "yellow"),
            marker=".",
            alpha=0.2,
            s=10,
        )
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Probability")
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    safe_desc = description.strip().replace(" ", "_")
    filepath = os.path.join(output_dir, f"convergence_{safe_desc}.png")
    plt.savefig(filepath)
    print(f"Annealing convergence plot saved to '{filepath}'")


def plot_pso_convergence(description: str, gbest_cost_history: List[float]) -> None:
    """
    Plots the convergence of the global best cost for Particle Swarm Optimization.
    """
    output_dir = "output/pso_convergence"
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    plt.figure(figsize=(15, 8))
    plt.plot(
        range(len(gbest_cost_history)),
        gbest_cost_history,
        label="Global Best Cost",
        color=get_color("mocha", "green"),
        linewidth=1.5,
    )

    plt.title(f"PSO Convergence for {description.strip()}", fontsize=16)
    plt.xlabel("Iteration")
    plt.ylabel("Cost (Objective Value)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    safe_desc = description.strip().replace(" ", "_")
    filepath = os.path.join(output_dir, f"convergence_{safe_desc}.png")
    plt.savefig(filepath)
    print(f"PSO convergence plot saved to '{filepath}'")
