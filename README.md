# ipysensitivityprofiler

Jupyter Widgets for visualizing local sensitivities of vectorized functions with signature `y = f(x)` where `x,y` are arrays.

--- 
## Installation

```
pip install ipysensitivityprofiler
```

--- 
## Example 

Example notebooks are avilable for download on [GitHub]() and hosted [binder.org](). 

--- 
## Documentation 

Documentation is available [here](TODO) (generated using [`sphinx`](https://www.sphinx-doc.org/en/master/))

--- 
## Usage

```
import numpy as np
import ipysensitivityprofiler as isp

def quadratic1(x):
    """y = x[0]**2 + x[2]**2 + x[0]*x[1]"""
    return (np.prod(x, axis=1) + np.power(x, 2).sum(axis=1))

def quadratic2(x):
    """y = 10 + x[0]**2 + x[2]**2 + x[0]*x[1]"""
    return (10 - 2 * np.prod(x, axis=1) + np.power(x, 2).sum(axis=1))

isp.profiler(
    models=[quadratic1, quadratic2], 
    xmin=[0, 0],
    xmax=[2, 1],
    ymin=[0],
    ymax=[20],
    x0=[1.5, 0.75],
    resolution=10_000, 
    xlabels=["x1", "x2"],
    ylabels=["y"],
)
```

![](docs/pics/example_usage.gif)

--- 
## License
Distributed under the terms of the MIT License.
