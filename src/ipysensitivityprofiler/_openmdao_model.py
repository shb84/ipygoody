"""Openmdao Model.

Module in charge of interfacing with an openmdao model.
"""

from functools import wraps
from importlib.util import find_spec
from typing import Any, Callable, List, Optional, Tuple, TypeAlias

import numpy as np

from ._model import Profiler, profiler
from ._view import DEFAULT_RESOLUTION, DEFAULT_WIDTH

_has_openmdao = True if find_spec("openmdao") else False


if _has_openmdao:
    from openmdao.api import Problem
    from openmdao.utils.units import convert_units

OpenMDAOProblem: TypeAlias = "Problem"


def requires_openmdao(func: Callable) -> Callable:
    """Return error if matplotlib not installed."""

    @wraps(func)
    def wrapper(*args: list, **kwargs: dict) -> Any:  # noqa: ANN401
        if _has_openmdao:
            return func(*args, **kwargs)
        raise ValueError("OpenMDAO is not installed.")

    return wrapper


@requires_openmdao
def openmdao_profiler(
    problem: OpenMDAOProblem,
    inputs: List[Tuple[str, float, float, Optional[str]]],
    outputs: List[Tuple[str, float, float, Optional[str]]],
    defaults: Optional[List[Tuple[str, float, Optional[str]]]] = None,
    resolution: int = DEFAULT_RESOLUTION,
    width: int = DEFAULT_WIDTH,
    height: Optional[int] = None,
) -> Profiler:
    """Create openmdao profiler for specified input/output labels.

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
