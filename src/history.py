import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils.ploter import config_fig, savefig
from utils.file_explorer import MODELS_DIR, DATA_HIST_DIR, PLOTS_HIST_DIR, makedir

def history() -> None:
    model_dir = os.path.join(MODELS_DIR, sys.argv[1])
    data_hist_dir = os.path.join(model_dir, DATA_HIST_DIR)

    # Load history
    hists = []
    for hist_file in os.listdir(data_hist_dir):
        hists.append(np.load(os.path.join(data_hist_dir, hist_file), allow_pickle="TRUE").item())

    # Make metric plots
    metrics = list(hists[0].keys())
    metrics = metrics[:(len(metrics) // 2)]
    for metric_name in metrics:
        info = describe_metric(hists, metric_name)

        # Set loss plot params
        is_loss = metric_name == metrics[0]
        ylim = None if is_loss else (0, 1.05)
        min_max_i = 1 if is_loss else 2

        config_fig("epoch [-]", f"{metric_name} [-]", ylim)
        plt.plot(info[0][0], label="trn (%.2g $\pm$ %.2g)" % info[0][min_max_i])
        plt.plot(info[1][0], label="val (%.2g $\pm$ %.2g)" % info[1][min_max_i])
        savefig(makedir(os.path.join(model_dir, PLOTS_HIST_DIR, f"{metric_name}.png")))

def describe_metric(hists: list[dict], metric_name: str) -> tuple[tuple[np.ndarray, tuple[float, float], tuple[float, float]], tuple[np.ndarray, tuple[float, float], tuple[float, float]]]:
    metric = [h[metric_name] for h in hists]
    val_metric = [h[f"val_{metric_name}"] for h in hists]
    return describe(metric), describe(val_metric)

def describe(metric_hists: list[list[float]]) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    mean = np.mean(metric_hists, axis=0)
    std = np.std(metric_hists, axis=0)
    min_i = np.argmin(mean)
    max_i = np.argmax(mean)
    return mean, (mean[min_i], std[min_i]), (mean[max_i], std[max_i])

if __name__ == "__main__":
    history()
