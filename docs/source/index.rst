.. ipysensitivityprofiler documentation master file, created by
   sphinx-quickstart on Sun Jan 23 10:35:00 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ipysensitivityprofiler's documentation!
==================================================

Jupyter Widgets for visualizing local sensitivities of callable Python functions in a notebook. 

What is a sensitivity profile? 
------------------------------
A local sensitivity profile is the trace of a function obtained by holding all dimensions fixed but one, 
as shown below. It can be thought of as the intersection of a cartesian plane (in which only one input is  
changing) and the response surface of interest.  

.. collapse:: show code

   .. code-block:: python 

      import numpy as np
      import matplotlib.pyplot as plt
      from mpl_toolkits.mplot3d import Axes3D


      # Define equation 
      def f(x): 
         return -0.1 * x[0] ** 3 - 0.5 * x[1] ** 2

      # Point about which to evaluate sensitivities
      x0 = np.array([[1], [1]])
      y0 = f(x0)

      # Define bounds of design space
      lb = [-5, -5]  # x1_min, x2_min
      ub = [ 5,  5]  # x1_max, x2_max

      # Grid coordinates per dimension (for plotting response surface)
      resolution = 100
      x1 = np.linspace(lb[0], ub[0], resolution).reshape((1, -1)) 
      x2 = np.linspace(lb[0], ub[0], resolution).reshape((1, -1))
      X1, X2 = np.meshgrid(x1, x2)
      x = np.concat([X1.reshape((1, -1)), X2.reshape((1, -1))])  # flatten grid
      y = f(x)  # evaluate points 
      Y = y.reshape(X1.shape)  # reshape grid 

      # Plot response surface 
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.plot_surface(X1, X2, Y, alpha=0.25) 

      # Plot profile along x1 
      x = np.concatenate([x1, x2])
      x[1, :] = x0[1]
      y = f(x)
      ax.plot(x[0], x[1], y, alpha=1, color='red', linewidth=2)

      # Plot profile along x2 
      x = np.concatenate([x1, x2])
      x[0, :] = x0[0]
      y = f(x)
      ax.plot(x[0], x[1], y, alpha=1, color='blue', linewidth=2)

      # Plot point about sensitivities are evaluated
      ax.plot(x0[0], x0[1], y0, "ko")
      ax.set_xlabel("x1")
      ax.set_ylabel("x2")
      ax.set_zlabel("y")
      plt.show()

.. image:: ../pics/slices.png

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Example Usage
-------------

.. collapse:: show code

   .. code-block:: python 

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

.. image:: ../pics/example_usage.gif

Audience
--------

The present library is intended for engineers who rely on modeling and simulation 
to make engineering design decisions. For example, developing physics-based or 
empirical models (e.g. ols, neural net) to make predictions about some system of interest 
or run optimization on it. 

Use Case 
--------

There are at least two use-cases for why such a widget is helpful.  

The first is debugging models. The ability to quickly interrogate the model(s) and 
get instantaneous feedback goes a long way in spotting obviously wrong trends early 
on. This is especially helpful when developing physics-based models, before using them 
for optimization or design decision-making. It's sometimes easier to look at trends 
to understand what is going on, than infer the issue from error stack messages.  

The second use-case is robust design. Upon convergence, the design team might be 
interested in understanding how system performance would change if the design was 
perturbed away from nominal. This could be the result of noise in the process, requirement
changes down the road, or some expected operating uncertainty. Perturbing inputs allows 
engineers to verify whether system outputs would stay within desired limits based on their model(s). 

Data Structures
---------------

The response :math:`f` can be any callable Python function that maps :math:`\boldsymbol{x}` to :math:`\boldsymbol{y}`, 
provided it is vectorized and adopts the following signature: 

.. math::

   \boldsymbol{y} = f(\boldsymbol{x}) 

where :math:`\boldsymbol{x}` and :math:`\boldsymbol{y}` are multidimensional arrays defined below, in which 
:math:`n_x` is the number of inputs, :math:`n_y` is the number of outputs, and :math:`m` is the number of examples: 

.. math::

   \boldsymbol{x} 
   =
   \left(
   \begin{matrix}
   x_1^{(1)} & \dots & x_1^{(m)} \\
   \vdots & \ddots & \vdots \\
   x_{n_x}^{(1)} & \dots & x_{n_x}^{(m)} \\
   \end{matrix}
   \right)
   \in 
   \mathbb{R}^{n_x \times m}
   \qquad 
   \boldsymbol{y} 
   =
   \left(
   \begin{matrix}
   y_1^{(1)} & \dots & y_1^{(m)} \\
   \vdots & \ddots & \vdots \\
   y_{n_y}^{(1)} & \dots & y_{n_y}^{(m)} \\
   \end{matrix}
   \right)
   \in 
   \mathbb{R}^{n_y \times m}

Limitations
-----------

This library relies on interactivity; hence, models must be fast. Concretely, the model must be 
able to evaluate thousands of datapoints simultaneously, on the order of milliseconds. This is a 
non-issue when the model at hand is some empirical regressionfor example, or evem some first-order 
physics-based model. 

The other limitation is screen realestate. This library is helpful for understanding how multiple 
responses and factors interact but, beyond a certain point, the number of inputs and outputs might 
be so big that human become overwhelmed with information and screen realestate runs out. Hence, 
this library is best suited for targeted studies on a subspace of a larger problem.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

API
---

.. automodule:: ipysensitivityprofiler
   :members:
