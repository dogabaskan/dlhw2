from typing import List, Union, Dict
import numpy as np
from dataclasses import dataclass, field
import plotly.graph_objects as go
import ipywidgets as widgets
import json
import os
from itertools import chain
import plotly.express as px


@dataclass
class Logger():
    """Logger for rendering and logging metrics"""

    epoch_train_accuracy: List[float] = field(default_factory=list)
    epoch_eval_accuracy: List[float] = field(default_factory=list)
    iter_train_accuracy: List[float] = field(default_factory=list)
    iter_train_loss: List[float] = field(default_factory=list)
    smooth_window_len: int = 250
    verbose: bool = True
    live_figure_update: bool = False

    def __post_init__(self):
        self.iteration_figure = self._make_figure(
            "iteration", "Train Iterations", ["accuracy", "nll loss"])
        self.epoch_figure = self._make_figure("epoch", "Per Epoch Accuracies", ["train", "eval"])

    def reset(self) -> None:
        self.epoch_train_accuracy = []
        self.epoch_eval_accuracy = []
        self.iter_train_accuracy = []
        self.iter_train_loss = []

    def render(self) -> widgets.HBox:
        """ Display iteration and epoch figures
        Returns:
            widgets.HBox: Horizontal Box of figures
        """
        return widgets.HBox(
            [self.iteration_figure, self.epoch_figure]
        )

    @staticmethod
    def _make_figure(x_axis_name: str, title: str, trace_names: List[str]) -> go.FigureWidget:
        """ Generate scatter plot
        Args:
            x_axis_name (str): Name of the x axis
            title (str): Title of the plot
            trace_names (List[str]): Legend names
        Returns:
            go.FigureWidget: Scatter figure
        """
        fig = go.FigureWidget()
        fig.update_layout(dict(
            template="none",
            width=500,
            xaxis=dict(title=x_axis_name),
            title=title
        ))
        for name in trace_names:
            fig.add_trace(go.Scatter(x=[], y=[], name=name, mode="markers+lines"))
        return fig

    def log_iteration(self, epoch: int, iteration: int) -> None:
        """ Write or render iteration specific metrics
        Args:
            epoch (int): Epoch number
            iteration (int): Iteration number
        """
        if iteration % self.smooth_window_len == 0:
            if self.verbose:
                print("Epoch: {}, Iteration: {}, Train loss: {:.4f}, Train acc: {:.4f}".format(
                    epoch,
                    iteration,
                    np.mean(self.iter_train_loss[-self.smooth_window_len:]),
                    np.mean(self.iter_train_accuracy[-self.smooth_window_len:])))
            if self.live_figure_update:
                x_axis = np.arange(len(self.iter_train_accuracy),
                                   step=self.smooth_window_len) * self.smooth_window_len
                self.iteration_figure.data[0].x = x_axis
                self.iteration_figure.data[0].y = self.smooth(self.iter_train_accuracy)
                self.iteration_figure.data[1].x = x_axis
                self.iteration_figure.data[1].y = self.smooth(self.iter_train_loss)

    def smooth(self, array: np.ndarray) -> np.ndarray:
        """ Smoothing function with stride
        Args:
            array (np.ndarray): 1D array of metric to smooth
        Returns:
            np.ndarray: Smoothed array
        """
        return np.convolve(array, np.ones(self.smooth_window_len)/self.smooth_window_len, mode="valid")[::self.smooth_window_len]

    def log_epoch(self, epoch: int) -> None:
        """ Write or render epoch specific metrics
        Args:
            epoch (int): Epoch number
        """
        if self.verbose:
            print("- " * 20)
            print("Epoch: {}, Train acc: {}, Eval acc: {}".format(
                epoch,
                self.epoch_train_accuracy[-1],
                self.epoch_eval_accuracy[-1]))

        if self.live_figure_update:
            x_axis = np.arange(len(self.epoch_train_accuracy))
            self.epoch_figure.data[0].x = x_axis
            self.epoch_figure.data[0].y = self.epoch_train_accuracy
            self.epoch_figure.data[1].x = x_axis
            self.epoch_figure.data[1].y = self.epoch_eval_accuracy

    @staticmethod
    def render_confusion_matrix(confusion_matrix: np.ndarray) -> go.FigureWidget:
        """ Plot confusion matrix
        Args:
            confusion_matrix (np.ndarray): 2D Confusion matrix array
        Returns:
            go.FigureWidget: Heatmap figure
        """
        fig = go.FigureWidget()
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            hoverongaps=False))
        fig.update_layout(dict(
            template="none",
            width=500,
            xaxis=dict(title="True Labels"),
            yaxis=dict(title="Predicted Labels", autorange="reversed"),
            title="Confusion Matrix"
        ))
        return fig

    def save_logs(self, path: str) -> None:
        """ Save logs to disk as a Json file

        Args:
            path (str): Log file path
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as write_obj:
            json.dump(dict(
                train_accuracy=[self._map_to_item(item)
                                for item in self.epoch_train_accuracy],
                eval_accuracy=[self._map_to_item(item) for item in self.epoch_eval_accuracy]),
                write_obj)

    @staticmethod
    def _map_to_item(value: Union[float, int, np.float32, np.float64]) -> float:
        """ Map values to float

        Args:
            value (Union[float, int, np.float32, np.float64]): Numpy floats, int or float

        Returns:
            float: Python float
        """
        if isinstance(value, (np.float32, np.int32, np.float64, np.float32)):
            return float(value.item())
        return float(value)

    @staticmethod
    def compare(experiment_path_dict: Dict[str, str]) -> go.FigureWidget:
        """ Return a comparison plot

        Args:
            experiment_path_dict (Dict[str, str]): Dictionary of experiment name and its path

        Returns:
            go.FigureWidget: Figure that contains the plots
        """
        plotly_colors = px.colors.qualitative.Plotly
        fig = Logger._make_figure("epoch", "Comparison", [])
        fig.update_layout(dict(width=800))
        for exp_index, (exp_name, exp_path) in enumerate(experiment_path_dict.items()):
            with open(exp_path, "r") as read_obj:
                exp_data = json.load(read_obj)
            color = plotly_colors[exp_index % len(plotly_colors)]
            for key, data in exp_data.items():
                fig.add_trace(go.Scatter(
                    x=np.arange(len(data)),
                    y=data,
                    name=f"{exp_name}<br>{key}",
                    mode="markers+lines",
                    line=dict(color=color, dash=("dash" if key == "train_accuracy" else "dot")),
                    legendgroup=exp_name,
                ))
        return fig
