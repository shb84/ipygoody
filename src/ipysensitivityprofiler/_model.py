"""Model.

Module in charge of interfacing with data generating models. For
example, such a model could be a simple callable y = f(x) which takes in
a numpy array of a certain shape or a more involved openmdao model that
requires more effort to get data in and out.
"""

from typing import Any, Callable, List, Optional, Union

import ipywidgets as W
import numpy as np

from ._controller import Controller
from ._view import DEFAULT_RESOLUTION, DEFAULT_WIDTH, View


class Profiler(W.VBox):
    """Profiler Widget."""

    def __init__(self, view: View, controller: Controller, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.view = view
        self.controller = controller
        self.children = [view, controller]


def profiler(
    models: List[Callable],
    xmin: Union[Union[List[float], np.ndarray], np.ndarray],
    xmax: Union[List[float], np.ndarray],
    ymin: Union[List[float], np.ndarray],
    ymax: Union[List[float], np.ndarray],
    x0: Optional[Union[List[float], np.ndarray]] = None,
    resolution: int = DEFAULT_RESOLUTION,
    width: int = DEFAULT_WIDTH,
    height: Optional[int] = None,
    xlabels: Optional[List[str]] = None,
    ylabels: Optional[List[str]] = None,
) -> Profiler:
    """Return profiler for function with signature y = f(x) where x, y are
    numpy arrays of shape (-1, nx) and (-1, ny), respectively.

    Parameters
    ----------
    models: List[callable]
        List of callable functions to be evaluated
        in order to generated profiles. There can be
        different models of the same process (e.g.
        low-fidelity and high-fidelity model of same thing),
        but they must have the same inputs/outputs.

    xmin: Union[List[float], np.ndarray]
        Lower bounds of inputs.

    xmax: Union[List[float], np.ndarray]
        Upper bounds of inputs.

    ymin: Union[List[float], np.ndarray]
        Lower bounds of outputs.

    ymax: Union[List[float], np.ndarray]
        Upper bounds of outputs.

    x0: Union[List[float], np.ndarray]
       Defaults to use for initial x0 (red dot in plots).
       Default is None (which turns into mean of range).

    resolution: int, optional
        Line resolution. Default is 25 points.

    width: int, optional
        Width of each plot. Default is 300 pixels.

    height: int, optional
         Height of each plot. Default is None (match width).

    xlabels: List[str]
        Labels to use for inputs. Default is None (which becomes x1, x2, ...)

    ylabels: Union[List[float], np.ndarray]
         Labels to use for outputs. Default is None (which becomes y1, y2, ...)
    """
    if height is None:
        height = width

    nx = len(xmin)
    ny = len(ymin)

    if x0 is None:
        x0 = [0.5 * (xmin[i] + xmax[i]) for i in range(nx)]

    def evaluate(x: np.ndarray) -> np.ndarray:
        outputs = []
        for f in models:
            y = f(x.reshape(-1, nx)).reshape((-1, ny, 1))
            outputs.append(y)
        return np.concatenate(outputs, axis=2)

    view = View(
        predict=evaluate,
        xlabels=xlabels,
        ylabels=ylabels,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        x0=x0,
        width=width * len(xmin),  # total width
        height=height * len(ymin),  # total height
        resolution=resolution,
    )

    controller = Controller(view)

    return Profiler(view, controller)
