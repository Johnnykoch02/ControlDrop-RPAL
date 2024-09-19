import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def plot_results_depracated(dictionaries, params, y_labels, range_spc, file_name):
    num_params = len(params)
    x_values = list(range_spc)
    fig, axs = plt.subplots(num_params, 1, figsize=(8, 6 * num_params))
    for idx, param in enumerate(params):
        y_values = [d[param] for d in dictionaries]
        for jdx, arr in enumerate(y_values):
            axs[idx].plot(x_values, arr, label=y_labels[jdx])
        axs[idx].set_xlabel("Samples")
        axs[idx].set_ylabel(param.capitalize())
        handles, labels = axs[idx].get_legend_handles_labels()
        axs[idx].legend(handles, labels)
    plt.tight_layout()
    plt.show()
    fig.savefig(file_name)


def plot_results(dictionaries, params, y_labels, range_spc, file_name):
    num_params = len(params)
    x_values = list(range_spc)
    fig = make_subplots(rows=num_params, cols=1, vertical_spacing=0.1)
    for idx, param in enumerate(params):
        y_values = [d[param] for d in dictionaries]
        for jdx, arr in enumerate(y_values):
            fig.add_trace(
                go.Scatter(x=x_values, y=arr, name=y_labels[jdx]), row=idx + 1, col=1
            )
        fig.update_xaxes(title_text="Samples", row=idx + 1, col=1)
        fig.update_yaxes(title_text=param.capitalize(), row=idx + 1, col=1)
    fig.update_layout(height=600 * num_params, width=800)
    fig.write_image(file_name)


def print_dict(d, indent=0):
    if isinstance(d, dict):
        for k, v in d.items():
            t = "\t" * indent
            print(f"{t}{k}:")
            print_dict(v, indent + 1)
    else:
        if isinstance(d, tuple) or isinstance(d, list):
            for i in d:
                print_dict(i, indent + 1)
        else:
            print("\t" * indent, d.shape)


### DEMO ###
# plot_resultsz([model_performance[0], model_performance[1], model_performance[2]],
#              ['performance', 'history', 'other'], ['easy', 'medium', 'hard'], range(100), os.path.join(os.getcwd(), 'test.png'))
