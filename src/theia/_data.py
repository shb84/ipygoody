"""Data.

Module in charge of data structure from which sensitivity profiles are
plotted.
"""
import numpy as np
import traitlets as T
import traittypes as TT


class Data(T.HasTraits):
    """Automatically evaluate outputs whenever inputs change."""

    xlabels = T.List(allow_none=False)
    ylabels = T.List(allow_none=False)
    predict = T.Callable(allow_none=False)

    x = TT.Array(allow_none=False)
    y = TT.Array()

    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

        def update(change):
            self.y = self.predict(self.x)

        self.observe(update, "predict")

        T.dlink((self, "x"), (self, "y"), self.predict)

    @property
    def n_x(self):
        """Number of inputs."""
        return len(self.xlabels)

    @property
    def n_y(self):
        """Number of outputs."""
        return len(self.ylabels)

    @property
    def N(self):
        """Number of models (i.e. number of lines on plot)"""
        return self.y.shape[2]

    @T.validate("x")
    def _validate_x(self, proposal):
        x = proposal.value
        if x.ndim != 2 or x.shape[1] != self.n_x or x.dtype != np.float64:
            return x.astype(float).reshape(-1, self.n_x)
        assert x.ndim == 2  # require shape (m, n_x)
        assert x.shape[1] == self.n_x
        assert x.dtype == np.float64
        return x

    @T.validate("y")
    def _validate_y(self, proposal):
        y = proposal.value
        assert y.ndim == 3  # require shape (m, n_y, N)
        assert y.shape[1] == self.n_y
        assert y.dtype == np.float64
        return y
