def plot_final_wealth_distribution(
    successful_sims: pd.DataFrame, column: str, title: str, xlabel: str, filename: str
):
    import numpy as np
    import matplotlib.pyplot as plt

    data = successful_sims[column].clip(lower=1.0)

    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Simulations")

    # PATCH: handle the degenerate case where all values are identical
    if (data == data.iloc[0]).all():
        plt.bar([data.iloc[0]], [len(data)], width=data.iloc[0]*0.05)
        plt.xscale('log')
    else:
        plt.hist(data, bins=30, log=True)
        plt.xscale('log')

    plt.tight_layout()
    plt.savefig(filename)
    # Do not close the figure, keep it open for interactive display