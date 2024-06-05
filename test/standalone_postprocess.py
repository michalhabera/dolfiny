def plot_convergence(jsonfile, title):
    import itertools
    import json

    import matplotlib.colors
    import matplotlib.pyplot as plt
    import numpy as np

    with open(jsonfile) as file:
        mi = json.load(file)

    fig, ax1 = plt.subplots()
    ax1.set_title(title, fontsize=12)
    ax1.set_xlabel(r"$\log (N)$", fontsize=12)
    ax1.set_ylabel(r"$\log (e)$", fontsize=12)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()

    markers = itertools.cycle(["o", "s", "x", "h", "p", "+"])
    colours = itertools.cycle(matplotlib.colors.TABLEAU_COLORS)
    for method, info in mi.items():
        marker = next(markers)
        colour = next(colours)
        lstyles = itertools.cycle(["-", "--", ":", "-."])
        for l2key, l2value in info["l2error"].items():
            lstyle = next(lstyles)
            n = np.log10(np.fromiter(l2value.keys(), dtype=float))
            e = np.log10(np.fromiter(l2value.values(), dtype=float))
            label = method + " (" + l2key + ")"
            ax1.plot(
                n,
                e,
                color=colour,
                linestyle=lstyle,
                marker=marker,
                linewidth=0.5,
                markersize=4,
                label=label,
            )
    ax1.legend(
        loc="lower left", ncol=len(mi), fontsize=4, markerscale=0.33, edgecolor="w", mode="expand"
    )

    # put order indication
    [xmin, xmax], [_, ymax] = ax1.get_xlim(), ax1.get_ylim()
    for k in [0, 1, 2, 3]:
        lx = xmin + (xmax - xmin) * np.array([0.8, 1.0])
        ly = ymax - (xmax - xmin) * np.array([0.0, 0.2 * k])
        ax1.plot(lx, ly, "k", linewidth=1.0)
        tx = lx[0] + 0.8 * (lx[1] - lx[0])
        ty = ly[0] + 0.8 * (ly[1] - ly[0])
        ax1.text(tx, ty, k, fontsize=8, backgroundcolor="w", va="center")

    fig.savefig(jsonfile + ".pdf")
