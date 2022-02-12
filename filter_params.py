from os import name

from pyqtgraph.parametertree.parameterTypes import SliderParameter
from pyqtgraph.parametertree import Parameter
from PyQt5.QtCore import QObject, pyqtSignal

import numpy as np

### my classes ###
from filters import *
from misc_methods import register_my_param, MyFrame
# from bubble_process import *

# Notes:
# - Parent classes cannot hold attributes
# BUG: When using pyqtgraph save and restores,
# it only restores parameters states,
# not any custom methods you made
# filter object becomes a regular parameter object with the filter params...
# how 2 fix? Idk
# eyy fixed make custom Parameter objects 11/8/21


class Filter(Parameter, QObject):
    swap_filter = pyqtSignal(str, str)

    def __init__(self, **opts):
        opts['removable'] = True
        # opts['context'] = ['Move Up', 'Move Down']
        super().__init__(**opts)

    def contextMenu(self, direction):
        self.swap_filter.emit(self.name(), direction)

    def process(self, frame):
        return frame

    def annotate(self, frame):
        return frame

    def __repr__(self):
        msg = self.opts['name'] + ' Filter'
        for c in self.childs:
            msg += f'\n{c.name()}: {c.value()}'
        return msg


@register_my_param
class Threshold(Filter):
    # cls_type register name with this class as a Parameter
    cls_type = 'ThresholdFilter'

    def __init__(self, **opts):
        # if opts['type'] is not specified here,
        # type will be filled in during saveState()
        # opts['type'] = self.cls_type

        # only set these params not passed params already
        if 'name' not in opts:
            opts['name'] = 'Threshold'

        if 'lower' not in opts:
            lower = 0
        else:
            lower = opts['lower']

        if 'children' not in opts:
            opts['children'] = [
                {
                    'name': 'Toggle',
                    'type': 'bool',
                    'value': opts.get('toggle', True)
                },
                {
                    'name': 'Thresh Type',
                    'type': 'list',
                    'value': 'thresh',
                    'limits': ['thresh', 'inv thresh', 'otsu', 'adaptive'],
                },
                {
                    'name': 'Upper',
                    'type': 'slider',
                    'value': 255,
                    'limits': (0, 255)
                },
                {
                    'name': 'Lower',
                    'type': 'slider',
                    'value': lower,
                    'limits': (0, 255)
                },
            ]
        super().__init__(**opts)

    def process(self, frame):
        frame = frame.view(MyFrame)

        return my_threshold(
            frame=frame,
            thresh=self.child('Lower').value(),
            maxval=self.child('Upper').value(),
            type=self.child('Thresh Type').value(),
        )


@register_my_param
class Normalize(Filter):
    # cls_type register name with this class as a Parameter
    cls_type = 'NormalizeFilter'

    def __init__(self, **opts):
        # if opts['type'] is not specified here,
        # type will be filled in during saveState()
        # opts['type'] = self.cls_type

        # only set these params not passed params already
        if 'name' not in opts:
            opts['name'] = 'Normalize'

        if 'children' not in opts:
            opts['children'] = [{
                'name': 'Toggle',
                'type': 'bool',
                'value': opts.get('toggle', True)
            }, {
                'title': 'Use Frame to Norm',
                'name': 'set_norm',
                'type': 'action',
            }]
        super().__init__(**opts)

        self.child('set_norm').sigActivated.connect(self.set_normalized)

        # used to subtract new frames to equalize lighting
        self.norm_frame = None
        self.init_norm = False

    def set_normalized(self):
        self.init_norm = True

    def process(self, frame):
        # is initializing norm frame
        if self.init_norm:
            # average the whole frame
            self.mean_frame = np.zeros(frame.shape, np.uint8)
            self.mean_frame[:] = frame.mean()
            # get the normalization frame so that
            # subsequent frame's lighting can be
            # closer to the mean
            # probably blur frame here before subtracting
            self.norm_frame = frame - self.mean_frame
            self.init_norm = False

        if self.norm_frame is not None:
            return MyFrame(frame - self.norm_frame)
        else:
            return frame


@register_my_param
class Contrast(Filter):
    # cls_type here to allow main_params.py to register this class as a Parameter
    cls_type = 'ContrastFilter'

    def __init__(self, **opts):
        # if opts['type'] is not specified here,
        # type will be filled in during saveState()
        # opts['type'] = self.cls_type

        # only set these params not passed params already
        if 'name' not in opts:
            opts['name'] = 'Contrast'

        if 'children' not in opts:
            opts['children'] = [
                {
                    'name': 'Toggle',
                    'type': 'bool',
                    'value': opts.get('toggle', True)
                },
                {
                    'title': 'Clip Limit',
                    'name': 'clip_lim',
                    'type': 'slider',
                    'value': opts.get('clip_lim', 2),
                    'step': 0.1,
                    'limits': (0, 20)
                },
                {
                    'title': 'Tile Size',
                    'name': 'tile_size',
                    'type': 'slider',
                    'value': opts.get('tile_size', 5),
                    'limits': (1, 255)
                },
            ]
        super().__init__(**opts)

    def process(self, frame):
        return my_contrast(
            frame=frame.cvt_color('gray'),
            clip_limit=self.child('clip_lim').value(),
            tile_size=self.child('tile_size').value(),
        )


@register_my_param
class Dilate(Filter):
    cls_type = 'DilateFilter'

    def __init__(self, **opts):
        # opts['type'] = self.cls_type

        if 'name' not in opts:
            opts['name'] = 'Dilate'

        if 'children' not in opts:
            opts['children'] = [
                {
                    'name': 'Toggle',
                    'type': 'bool',
                    'value': opts.get('toggle', True)
                },
                {
                    'name': 'Iterations',
                    'type': 'int',
                    'value': 1,
                    'limits': (0, 50)
                },
            ]
        super().__init__(**opts)

    def process(self, frame):
        return MyFrame(
            my_dilate(frame=frame,
                      iterations=self.child('Iterations').value()),
            frame.colorspace)


@register_my_param
class HoughCircle(Filter):
    cls_type = 'HoughFilter'

    def __init__(self, **opts):
        # opts['type'] = self.cls_type

        if 'name' not in opts:
            opts['name'] = 'Hough'

        if 'children' not in opts:
            opts['children'] = [
                {
                    'name': 'Toggle',
                    'type': 'bool',
                    'value': opts.get('toggle', True)
                },
                {
                    'name': 'dp',
                    'title': 'DP',
                    'type': 'float',
                    'step': 0.1,
                    'value': 1.2,
                    'limits': (0, 500),
                },
                {
                    'name': 'min_dist',
                    'title': 'Min Dist',
                    'type': 'int',
                    'value': 100,
                    'limits': (1, 10000),
                },
            ]
        super().__init__(**opts)

    def process(self, frame):
        return my_hough(frame,
                        dp=self.child('dp').value(),
                        min_dist=self.child('min_dist').value())


@register_my_param
class Erode(Filter):
    cls_type = 'ErodeFilter'

    def __init__(self, **opts):
        # opts['type'] = self.cls_type

        if 'name' not in opts:
            opts['name'] = 'Erode'

        if 'children' not in opts:
            opts['children'] = [
                {
                    'name': 'Toggle',
                    'type': 'bool',
                    'value': opts.get('toggle', True)
                },
                {
                    'name': 'Iterations',
                    'type': 'int',
                    'value': 1,
                    'limits': (0, 50)
                },
            ]
        super().__init__(**opts)

    def process(self, frame):
        return my_erode(frame, iterations=self.child('Iterations').value())


@register_my_param
class Invert(Filter):
    cls_type = 'InvertFilter'

    def __init__(self, **opts):
        # opts['type'] = self.cls_type

        if 'name' not in opts:
            opts['name'] = 'Invert'

        if 'children' not in opts:
            opts['children'] = [
                {
                    'name': 'Toggle',
                    'type': 'bool',
                    'value': opts.get('toggle', True)
                },
            ]
        super().__init__(**opts)

    def process(self, frame):
        return my_invert(frame)


@register_my_param
class Edge(Filter):
    cls_type = 'EdgeFilter'

    def __init__(self, **opts):
        # opts['type'] = self.cls_type

        if 'name' not in opts:
            opts['name'] = 'Canny Edge'

        if 'children' not in opts:
            opts['children'] = [
                {
                    'name': 'Toggle',
                    'type': 'bool',
                    'value': opts.get('toggle', True)
                },
                SliderParameter(name='Thresh1', value=1, limits=(0, 255)),
                SliderParameter(name='Thresh2', value=1, limits=(0, 255)),
            ]
        super().__init__(**opts)

    def process(self, frame):
        return canny_edge(
            frame,
            thresh1=self.child('Thresh1').value(),
            thresh2=self.child('Thresh2').value(),
        )


@register_my_param
class Blur(Filter):
    cls_type = 'BlurFilter'

    def __init__(self, **opts):
        # opts['type'] = self.cls_type

        if 'name' not in opts:
            opts['name'] = 'Blur'

        if 'children' not in opts:
            opts['children'] = [
                {
                    'name': 'Toggle',
                    'type': 'bool',
                    'value': True
                },
                {
                    'name': 'Type',
                    'type': 'list',
                    'value': 'Gaussian',
                    'limits': ['Gaussian', 'Blur', 'Median'],
                },
                {
                    'name': 'Radius',
                    'type': 'int',
                    'value': 1,
                    'limits': (0, 100)
                },
                {
                    'name': 'Iterations',
                    'type': 'int',
                    'value': 1,
                    'limits': (0, 50)
                },
            ]

        super().__init__(**opts)

    def process(self, frame):
        return my_blur(
            frame,
            radius=self.child('Radius').value(),
            iterations=self.child('Iterations').value(),
            view=self.child('Type').value(),
        )
