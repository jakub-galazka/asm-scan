import matplotlib.pyplot as plt

def config_fig(xlabel: str, ylabel: str, ylim: tuple[float] = None, style: str = "seaborn"):
    plt.style.use(style)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if bool(ylim):
        plt.ylim(ylim)

def savefig(filepath: str, legend: bool = True):
    if legend:
        plt.legend()
    plt.savefig(filepath)
    plt.clf()
