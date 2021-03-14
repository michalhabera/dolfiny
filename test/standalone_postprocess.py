

def plot_convergence(jsonfile, title):

    import matplotlib.pyplot
    import matplotlib.colors
    import itertools
    import numpy
    import json

    with open(jsonfile, 'r') as file:
        mi = json.load(file)

    fig, ax1 = matplotlib.pyplot.subplots()
    ax1.set_title(title, fontsize=12)
    ax1.set_xlabel(r'$\log (N)$', fontsize=12)
    ax1.set_ylabel(r'$\log (e)$', fontsize=12)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()
    markers = itertools.cycle(['o', 's', 'x', 'h', 'p', '+'])
    colours = itertools.cycle(matplotlib.colors.TABLEAU_COLORS)
    for method, info in mi.items():
        marker = next(markers)
        colour = next(colours)
        lstyles = itertools.cycle(['-', '--', ':'])
        for l2key, l2value in info["l2error"].items():
            lstyle = next(lstyles)
            n = numpy.log10(numpy.fromiter(l2value.keys(), dtype=float))
            e = numpy.log10(numpy.fromiter(l2value.values(), dtype=float))
            label = method + " (" + l2key + ")" if l2key == 'u' else None
            ax1.plot(n, e, color=colour, linestyle=lstyle, marker=marker, linewidth=1, markersize=5, label=label)
    ax1.legend(loc='lower left')
    fig.savefig(jsonfile + ".pdf")
