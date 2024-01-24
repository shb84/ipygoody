"""Controller.

Module contains widgets, such as slider bars, to control model and view.
"""
from functools import partial

import ipywidgets as W

from ._view import View


def _update_ylim(view: View, index: int, proposal: dict):
    if proposal.new != proposal.old:
        ymin = view.ymin.tolist()
        ymax = view.ymax.tolist()
        ymin[index] = proposal.new[0]
        ymax[index] = proposal.new[1]
        with view.hold_sync():
            view.ymax = ymax
            view.ymin = ymin


def _update_xlim(view: View, index, proposal: dict):
    if proposal.new != proposal.old:
        xmin = view.xmin.tolist()
        xmax = view.xmax.tolist()
        xmin[index] = proposal.new[0]
        xmax[index] = proposal.new[1]
        with view.hold_sync():
            view.xmax = xmax
            view.xmin = xmin


def _update_x0(view: View, index: int, proposal: dict):
    if proposal.new != proposal.old:
        x0 = view.x0.tolist()
        x0[0][index] = proposal.new
        view.x0 = x0


def _create_sliders(view: View):
    sliders = []
    for i in range(view.data.n_x):
        slider = W.FloatSlider(
            value=view.x0[0, i],
            min=view.xmin[i],
            max=view.xmax[i],
            step=(view.xmax[i] - view.xmin[i]) / view.resolution,
            description=view.xlabels[i],
        )
        sliders.append(slider)
    return sliders


def _create_range_sliders(view: View):
    sliders = {"x": [], "y": []}

    for i in range(view.data.n_y):
        slider = W.FloatRangeSlider(
            value=(view.ymin[i], view.ymax[i]),
            min=view.ymin[i],
            max=view.ymax[i],
            description=view.ylabels[i],
            step=(view.ymax[i] - view.ymin[i]) / 10_000,
            orientation="horizontal",
            readout=False,
        )
        sliders["y"].append(slider)

    for i in range(view.data.n_x):
        slider = W.FloatRangeSlider(
            value=(view.xmin[i], view.xmax[i]),
            min=view.xmin[i],
            max=view.xmax[i],
            description=view.xlabels[i],
            step=(view.xmax[i] - view.xmin[i]) / 10_000,
            orientation="horizontal",
            readout=False,
        )
        sliders["x"].append(slider)

    return sliders


class Controller(W.VBox):
    """Control panel for profiler."""

    def __init__(self, view: View, **kwargs):
        super().__init__(**kwargs)

        self.range_sliders = _create_range_sliders(view)

        for i, range_slider in enumerate(self.range_sliders["y"]):
            range_slider.observe(partial(_update_ylim, view, i), "value")

        for j, range_slider in enumerate(self.range_sliders["x"]):
            range_slider.observe(partial(_update_xlim, view, j), "value")

        self.sliders = _create_sliders(view)

        for i, slider in enumerate(self.sliders):
            slider.observe(partial(_update_x0, view, i), "value")

        self.children = [
            *self.sliders,
            *self.range_sliders["x"],
            *self.range_sliders["y"],
        ]
