"""Model.

Module in charge of interfacing with data generating models. For example,
such a model could be a simple callable y = f(x) which takes in a numpy
array of a certain shape or a more involved openmdao model that requires
more effort to get data in and out.
"""
import openmdao.api as om 
import numpy as np 
import ipywidgets as W 

from openmdao.utils.units import convert_units

from ._view import View, DEFAULT_WIDTH, DEFAULT_RESOLUTION
from ._controller import Controller


class Profiler(W.VBox): 

    def __init__(self, view: View, controller: Controller, **kwargs): 
        super().__init__(**kwargs)
        self.view = view 
        self.controller = controller
        self.children = [view, controller]


def profiler(
        models: list[callable], 
        xmin: list[float],
        xmax: list[float],
        ymin: list[float],
        ymax: list[float],
        x0: list[float] = None, 
        resolution: int = DEFAULT_RESOLUTION,
        width: int = DEFAULT_WIDTH,
        height: int = None,
        xlabels: list[str] = None,
        ylabels: list[str] = None,
) -> Profiler: 
    """Return profiler for function with signature y = f(x) 
    where x, y are numpy arrays of shape (-1, nx) and (-1, ny), respectively.
    
    Parameters
    ----------
    models: list[callable]
        List of callable functions to be evaluated 
        in order to generated profiles. There can be 
        different models of the same process (e.g. 
        low-fidelity and high-fidelity model of same thing), 
        but they must have the same inputs/outputs. 

    xmin: list[float]
        Lower bounds of inputs. 

    xmax: list[float]
        Upper bounds of inputs.

    ymin: list[float]
        Lower bounds of outputs. 

    ymax: list[float]
        Upper bounds of outputs.

    x0: list[float]
       Defaults to use for initial x0 (red dot in plots).
       Default is None (which turns into mean of range). 

    resolution: int, optional 
        Line resolution. Default is 25 points.  

    width: int, optional 
        Width of each plot. Default is 300 pixels.   

    height: int, optional 
         Height of each plot. Default is None (match width).   

    xlabels: list[str]
        Labels to use for inputs. Default is None (which becomes x1, x2, ...)

    ylabels: list[float]
         Labels to use for outputs. Default is None (which becomes y1, y2, ...)
    """
    if height is None: 
        height = width 

    nx = len(xmin) 
    ny = len(ymin) 

    if x0 is None: 
        x0 = [0.5 * (xmin[i] + xmax[i]) for i in range(nx)]

    def evaluate(x): 
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
        width=width * len(xmin),   # total width
        height=height * len(ymin), # total height
        resolution=resolution,
    )

    controller = Controller(view)

    return Profiler(view, controller)


def openmdao_profiler(
    problem: om.Problem,
    inputs: list[tuple[str, float, float, str]],
    outputs: list[tuple[str, float, float, str]],  
    defaults: dict = None,  
    resolution: int = DEFAULT_RESOLUTION,
    width: int = DEFAULT_WIDTH,
    height: int = None,
):
    """Create profiler of provided openmdao model and specified input/output labels.
    
    Parameters
    ----------
    inputs: list[tuple[str, float, float, str]]
        Inputs and associated bounds to display. 
        Format: [(name, min, max, units)]

    outputs: list[tuple[str, float, float, str]]
        Outputs and associated bounds to display. 
        Format: [(name, min, max, units)]

    defaults: dict[str, tuple[float, str, str | None]], optional 
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

    values = [] 
    for i, name in enumerate(x_labels): 
        if defaults is None: 
            value = 0.5 * (x_min[i] + x_max[i])
        else: 
            value = convert_units(
                val=defaults[name]["val"],
                old_units=defaults[name]["units"], 
                new_units=x_units[i], 
            )
        values.append(value)
    x0 = np.array(values)

    def evaluate(x: np.ndarray):
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