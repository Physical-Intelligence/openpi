import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/project/shihanzh/openpi/third_party/libero/scripts')
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, no_update
from qd_spatial import evaluate
from ribs.visualize import grid_archive_heatmap


def show_interactive_archive(archive):
    fig = plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=0, vmax=1, cmap="viridis")
    plt.tight_layout()

    def onclick(event):
        occupied, data = archive.retrieve_single([event.xdata, event.ydata])

        if occupied:
            print(
                f'Recorded objective: {data["objective"]}; Recorded measures: {data["measures"]}'
            )
            obj, meas = evaluate(params=[data["solution"]], ntrials=1, seed=42, video_logdir='interactive_vids')
        else:
            print("Archive cell not occupied")

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()


def _plotly_grid_archive_heatmap(archive, shape={'width': 600, 'height': 600}):
    x_dim, y_dim = archive.dims
    colors = np.full((y_dim, x_dim), np.nan)
    index_batch = archive.data("index")
    objective_batch = archive.data("objective")
    grid_index_batch = archive.int_to_grid_index(index_batch)
    colors[grid_index_batch[:, 1], grid_index_batch[:, 0]] = objective_batch

    x_bounds = archive.boundaries[0]
    y_bounds = archive.boundaries[1]

    fig = go.Figure(data=go.Heatmap(
        z=colors,
        x=x_bounds,
        y=y_bounds,
        colorbar= {
            "title": 'Ent. success'
        },
        colorscale='Viridis'
    ))
    fig.update_layout(**shape)
    fig.update_xaxes(title='Spread')
    fig.update_yaxes(title='Similarity')

    return fig


def host_interactive_archive(archive, port=8050):
    '''Similar to :func:`show_interactive_archive`, except it hosts the 
    interactive plot at localhost:<port> to allow generating the plot on a 
    remote machine and then viewing it on your local machine (e.g. if you only 
    have access to ssh). After configuring port forwarding between your local 
    machine and ``port`` on the remote machine, you will be able to view and 
    interact with the plot on your local machine's browser.

    Args:
        archive (GridArchive): Archive to be displayed.
        port (int): The port on which to display the plot.
    '''
    app = Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(id='archive-heatmap', figure=_plotly_grid_archive_heatmap(archive)),
        html.Div(id='dummy-output', style={'display': 'none'})  # hidden dummy output
    ], style={'display': 'flex', 'justifyContent': 'center'})
        
    @app.callback(
        Output('dummy-output', 'children'),
        Input("archive-heatmap", "clickData"),
    )
    def onclick(clickData):
        if clickData is None:
            return no_update
        
        occupied, data = archive.retrieve_single([clickData["points"][0]["x"], clickData["points"][0]["y"]])

        if occupied:
            print(
                f'Recorded objective: {data["objective"]}; Recorded measures: {data["measures"]}'
            )
            evaluate(params=[data["solution"]], ntrials=5, seed=42, video_logdir='interactive_vids')
        else:
            print("Archive cell not occupied")

        return None

    app.run(host="0.0.0.0", port=port, debug=True)
    

if __name__ == "__main__":
    with open(
        file="test_logs/scheduler_00000200.pkl",
        mode="rb",
    ) as f:
        archive = pkl.load(f).result_archive
        # show_interactive_archive(archive)
        host_interactive_archive(archive)
