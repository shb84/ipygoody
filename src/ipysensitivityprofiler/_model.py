"""Model.

Module in charge of interfacing with data generating models. For
example, such a model could be a simple callable y = f(x) which takes in
a numpy array of a certain shape or a more involved openmdao model that
requires more effort to get data in and out.
"""

from typing import Any, Callable, List, Optional, Tuple, Union

import ipywidgets as W
import numpy as np
import openmdao.api as om
from openmdao.utils.units import convert_units

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


def openmdao_profiler(
    problem: om.Problem,
    inputs: List[Tuple[str, float, float, Optional[str]]],
    outputs: List[Tuple[str, float, float, Optional[str]]],
    defaults: Optional[List[Tuple[str, float, Optional[str]]]] = None,
    resolution: int = DEFAULT_RESOLUTION,
    width: int = DEFAULT_WIDTH,
    height: Optional[int] = None,
) -> Profiler:
    """Create profiler of provided openmdao model and specified input/output
    labels.

    Parameters
    ----------
    inputs: List[Tuple[str, float, float, str | None]]
        Inputs and associated bounds to display.
        Format: [(name, min, max, units)]

    outputs: List[Tuple[str, float, float, str | None]]
        Outputs and associated bounds to display.
        Format: [(name, min, max, units)]

    defaults: List[Tuple[str, float, str | None]], optional
        Defaults to use for initial x0 (red dot in plots).
        Default is None (which turns into mean of range).
        Format: {name: (val, units)}

    resolution: int, optional
        Line resolution. Default is 25 points.

    width: int, optional
        Width of each plot. Default is 300 pixels.

    height: int, optional
         Height of each plot. Default is None (match width).
    """
    if height is None:
        height = width

    problem.model.options["num_nodes"] = len(inputs) * resolution
    problem.setup()

    x_labels = []
    x_min = []
    x_max = []
    x_units = []
    for name, lower, upper, units in inputs:
        x_labels.append(name)
        x_min.append(lower)
        x_max.append(upper)
        x_units.append(units)

    y_labels = []
    y_min = []
    y_max = []
    y_units = []
    for name, lower, upper, units in outputs:
        y_labels.append(name)
        y_min.append(lower)
        y_max.append(upper)
        y_units.append(units)

    n_x = len(x_labels)
    n_y = len(y_labels)
    m = n_x * resolution

    x_min = np.array(x_min)
    x_max = np.array(x_max)

    # Convert defaults to dictionary for lookup
    items = defaults if defaults else ()
    defaults = (
        {}
        if defaults is None
        else {item[0]: {"val": item[1], "units": item[2]} for item in items}
    )

    values = []
    for i, name in enumerate(x_labels):
        if name in defaults:
            value = convert_units(
                val=defaults[name]["val"],
                old_units=defaults[name]["units"],
                new_units=x_units[i],
            ).squeeze()
        else:
            value = 0.5 * (x_min[i] + x_max[i])
        values.append(value)
    x0 = np.array(values)

    def evaluate(x: np.ndarray) -> np.ndarray:
        for i, x_label in enumerate(x_labels):
            problem.set_val(x_label, x[:, i], x_units[i])
        problem.run_model()
        y = np.zeros((m, n_y, 1))
        for i, y_label in enumerate(y_labels):
            y[:, i, 0] = problem.get_val(y_label, y_units[i])
        return y.reshape((m, -1, 1))

    return profiler(
        models=[evaluate],
        xmin=x_min,
        xmax=x_max,
        ymin=y_min,
        ymax=y_max,
        x0=x0,
        resolution=resolution,
        width=width,
        height=height,
        xlabels=x_labels,
        ylabels=y_labels,
    )
