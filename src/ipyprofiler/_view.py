"""View.

Module contains widgets and data to render interactive sensivity
profiles.
"""
import pathlib as pl

import bqplot as bq
import ipywidgets as W
import numpy as np
import traitlets as T
import traittypes as TT

from ._data import Data
from ._utils import create_batches, create_grid, make_grid

DEFAULT_RESOLUTION = 25
DEFAULT_WIDTH = 300
DEFAULT_HEIGHT = DEFAULT_WIDTH


class View(W.Box):
    """Widget that displays grid of sensivity profiles."""

    # _view_module = T.Unicode('sensivity_profiler').tag(sync=True)
    # _view_module_version = T.Unicode('0.0.0').tag(sync=True)
    # _view_name = T.Unicode('SensivityProfiler').tag(sync=True)

    ##########
    # Traits #
    ##########

    predict = T.Callable(allow_none=False)

    xmin = TT.Array(allow_none=False)
    xmax = TT.Array(allow_none=False)

    ymin = TT.Array(allow_none=False)
    ymax = TT.Array(allow_none=False)

    width = T.Int(allow_none=True)
    height = T.Int(allow_none=True)

    resolution = T.Int(default_value=DEFAULT_RESOLUTION)

    x0 = TT.Array(allow_none=False)
    y0 = TT.Array(allow_none=False)

    xlabels = T.List(allow_none=False)
    ylabels = T.List(allow_none=False)

    data = T.Instance(klass=Data)
    grid = T.Instance(klass=W.GridspecLayout)

    _batches = T.List(allow_none=True)
    _lines = T.List(allow_none=True)
    _dots = T.List(allow_none=True)

    #################
    # Instantiation #
    #################

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        #########
        # Links #
        #########

        T.dlink((self, "x0"), (self, "y0"), lambda x0: self.predict(x0))
        T.link((self, "predict"), (self.data, "predict"))

        ##############
        # Initialize #
        ##############

        self._update_data()
        self._update_figs()
        self._update_lims()
        self._update_labels()

        #############
        # Observers #
        #############

        self.observe(self._update_lims, ["xmin", "xmax", "ymin", "ymax"])
        self.observe(self._update_data, ["x0", "xmin", "xmax", "resolution"])
        self.observe(self._update_figs, ["x0", "data"])
        self.observe(self._update_labels, ["xlabels", "ylabels"])

        self.children = [self.grid]

    ###############
    # Interactive #
    ###############

    def _update_data(self, *args, **kwargs):
        self.data.x = create_grid(
            x0=self.x0, xmin=self.xmin, xmax=self.xmax, resolution=self.resolution
        )

    def _update_figs(self, *args, **kwargs):
        with self.grid.hold_sync():
            for j in range(self.data.n_x):
                batch = self._batches[
                    j
                ]  # Data is a huge grid. These are indices of relevant subset.
                for i in range(self.data.n_y):
                    k = 0
                    for mark in self.grid[
                        i, j
                    ].marks:  # [line1, cursor1, line2, cursor2, ...]
                        if isinstance(mark, bq.marks.Lines):
                            line = mark
                            line.x = self.data.x[batch, j]
                            line.y = self.data.y[batch, i, k]
                        if isinstance(mark, bq.marks.Scatter):
                            dot = mark
                            dot.x = self.x0[0, j : j + 1]
                            dot.y = self.y0[0, i : i + 1, k]
                            k += 1

    def _update_lims(self, *args, **kwargs):
        with self.grid.hold_sync():
            for j in range(self.data.n_x):
                for i in range(self.data.n_y):
                    self.grid[i, j].axes[0].scale.min = self.xmin[j]
                    self.grid[i, j].axes[0].scale.max = self.xmax[j]
                    self.grid[i, j].axes[1].scale.min = self.ymin[i]
                    self.grid[i, j].axes[1].scale.max = self.ymax[i]

    def _update_labels(self):
        with self.grid.hold_sync():
            for j in range(self.data.n_x):
                for i in range(self.data.n_y):
                    self.grid[i, j].axes[0].label = self.xlabels[j]
                    self.grid[i, j].axes[1].label = self.ylabels[i]

    ############
    # Defaults #
    ############

    @T.default("x0")
    def _create_x0(self):
        x0 = 0.5 * (self.xmin + self.xmax).reshape((1, -1))
        return x0

    # @T.default('y0')
    # def _create_y0(self):
    #     y0 = self.predict(self.x0)
    #     return y0

    @T.default("xlabels")
    def _create_xlabels(self):
        n_x = self.xmin.size
        return [f"x{i}" for i in range(n_x)]

    @T.default("ylabels")
    def _create_ylabels(self):
        n_y = self.ymin.size
        return [f"y{i}" for i in range(n_y)]

    @T.default("data")
    def _create_data(self):
        data = Data(
            predict=self.predict,
            xlabels=self.xlabels,
            ylabels=self.ylabels,
            x=create_grid(
                x0=self.x0, xmin=self.xmin, xmax=self.xmax, resolution=self.resolution
            ),
        )
        return data

    @T.default("grid")
    def _create_view(self):
        grid = make_grid(
            self.data.n_x, self.data.n_y, self.data.N, self.width, self.height,
        )
        return grid

    @T.default("_batches")
    def _create_batches(self):
        batches = create_batches(self.data.n_x, self.resolution)
        return batches

    ############
    # Validate #
    ############

    @T.validate("width")
    def _validate_width(self, proposal):
        width = proposal.value
        if width is None:
            return DEFAULT_WIDTH * self.data.n_y
        return width

    @T.validate("height")
    def _validate_height(self, proposal):
        height = proposal.value
        if height is None:
            return DEFAULT_HEIGHT * self.data.n_y
        return height

    @T.validate("xlabels")
    def _validate_xlabels(self, proposal):
        xlabels = proposal.value
        if not xlabels:
            return self._create_xlabels()
        assert len(xlabels) == self.data.n_x
        return xlabels

    @T.validate("ylabels")
    def _validate_ylabels(self, proposal):
        ylabels = proposal.value
        if not ylabels:
            return self._create_ylabels()
        assert len(ylabels) == self.data.n_y
        return ylabels

    @T.validate("y0")
    def _validate_y0(self, proposal):
        array = proposal.value
        assert array.ndim == self.data.y.ndim
        return array.astype(np.float64)

    @T.validate("x0")
    def _validate_x0(self, proposal):
        array = proposal.value
        if array is None:
            return self._create_x0()
        array = array.reshape((1, -1))
        assert array.ndim == self.data.x.ndim
        return array.astype(np.float64)

    @T.validate("xmin")
    def _validate_xmin(self, proposal):
        array = proposal.value
        return array.astype(np.float64).ravel()

    @T.validate("xmax")
    def _validate_xmax(self, proposal):
        array = proposal.value
        return array.astype(np.float64).ravel()

    @T.validate("ymin")
    def _validate_ymin(self, proposal):
        array = proposal.value
        return array.astype(np.float64).ravel()

    @T.validate("ymax")
    def _validate_ymax(self, proposal):
        array = proposal.value
        return array.astype(np.float64).ravel()

    ###########
    # Methods #
    ###########

    # TODO: need to add white background to grid boxes, else pictures are transparent

    def save_png(self, xlabel: str, ylabel: str):
        """Save figure to PNG."""
        filename = f"profiler_{xlabel}_vs_{ylabel}.png"
        file = pl.Path(filename)
        j = self.xlabels.index(xlabel)
        i = self.ylabels.index(ylabel)
        figure = self.grid[i, j]

        def save_when_ready(data) -> None:
            file.write_bytes(data)

        figure.get_png_data(save_when_ready)
