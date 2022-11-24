import cv2
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

    def on_roi_updated(self, roi):
        print('roi updated')

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
        self.thresh_types = {
            'otsu': cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            'thresh': cv2.THRESH_BINARY,
            'inv thresh': cv2.THRESH_BINARY_INV,
        }

        # only set these params not passed params already
        if 'name' not in opts:
            opts['name'] = 'Threshold'

        if 'children' not in opts:
            opts['children'] = [
                {
                    'name': 'Toggle',
                    'type': 'bool',
                    'value': opts.get('toggle', True)
                },
                {
                    'title': 'Thresh Type',
                    'name': 'thresh_type',
                    'type': 'list',
                    'value': opts.get('type', 'otsu'),
                    'limits': list(self.thresh_types.keys()),
                },
                {
                    'title': 'Upper',
                    'name': 'upper',
                    'type': 'slider',
                    'value': 255,
                    'limits': (0, 255),
                    'visible': False
                },
                {
                    'title': 'Threshold',
                    'name': 'lower',
                    'type': 'slider',
                    'value': opts.get('lower', 0),
                    'limits': (0, 255)
                },
            ]
        super().__init__(**opts)

    def process(self, frame):
        frame = frame.view(MyFrame)

        thresh_type = self.thresh_types[self.child('thresh_type').value()]

        frame = frame.cvt_color('gray')
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('frame', frame)
        thresh_val, frame = cv2.threshold(frame,
                                          self.child('lower').value(),
                                          self.child('upper').value(),
                                          thresh_type)
        # cv2.imshow('frame after', frame)

        if thresh_type == cv2.THRESH_OTSU:
            print('Auto Thresh:', thresh_val)
            self.child('lower').setValue(thresh_val)
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        return MyFrame(frame, 'gray')


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
            }, {
                'title': 'View Norm',
                'name': 'view_norm',
                'type': 'action',
            }]
        super().__init__(**opts)

        self.child('set_norm').sigActivated.connect(self.set_normalized)
        self.child('view_norm').sigActivated.connect(self.view_norm)

        # used to subtract new frames to equalize lighting
        self.norm_frame = None
        self.init_norm = False
        self.saved_frame = None
        self.roi = None

    # this is doodoo
    # the original frame has to be of full size
    # for this to work
    def on_roi_updated(self, roi):
        self.roi = roi
        return super().on_roi_updated(roi)

    def set_normalized(self):
        self.init_norm = True

    def view_norm(self):
        cv2.imshow('Norm', self.saved_frame)

    def process(self, frame):
        # is initializing norm frame
        if self.init_norm:
            # average the whole frame
            self.saved_frame = frame
            self.mean_frame = np.zeros(frame.shape)
            self.mean_frame[:] = frame.mean()
            # get the normalization frame so that
            # subsequent frame's lighting can be
            # closer to the mean
            self.norm_frame = frame - self.mean_frame
            self.init_norm = False
            # cv2.imshow('norm', self.norm_frame)

        if self.norm_frame is not None:
            frame = frame.astype(np.int16) - self.norm_frame
            frame[frame > 255] = 255
            frame[frame < 0] = 0
            return MyFrame(np.uint8(frame))
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
                    'title': 'Brightness',
                    'name': 'brightness',
                    'type': 'slider',
                    'value': opts.get('brightness', 0),
                    'step': 1,
                    'limits': (-255, 256)
                },
                {
                    'title': 'Contrast',
                    'name': 'contrast',
                    'type': 'float',
                    'value': opts.get('contrast', 1),
                    'step': 0.1,
                    'limits': (0, 20)
                },
            ]
        super().__init__(**opts)

    def process(self, frame):
        frame = frame + self.child('brightness').value()
        frame = frame * self.child('contrast').value()
        frame[frame > 255] = 255
        frame[frame < 0] = 0
        return np.uint8(frame)


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
        return MyFrame(cv2.bitwise_not(frame))


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
                    'title': 'Type',
                    'name': 'type',
                    'type': 'list',
                    'value': opts.get('type', 'Gaussian'),
                    'limits': ['Gaussian', 'Blur', 'Median'],
                },
                {
                    'title': 'Radius',
                    'name': 'radius',
                    'type': 'int',
                    'value': opts.get('radius', 1),
                    'limits': (0, 100)
                },
                {
                    'title': 'Iterations',
                    'name': 'iterations',
                    'type': 'int',
                    'value': opts.get('iteration', 1),
                    'limits': (0, 50)
                },
            ]

        super().__init__(**opts)

    def process(self, frame):
        view = self.child('type').value()
        radius = self.child('radius').value()
        radius = radius * 2 + 1  # have only odd radius, needed for the blur kernel
        kernel = (radius, radius)
        for i in range(self.child('iterations').value()):
            if view == "Gaussian":
                frame = cv2.GaussianBlur(frame, kernel, cv2.BORDER_DEFAULT)
            elif view == "Median":
                frame = cv2.medianBlur(frame, radius)
            elif view == "Blur":
                frame = cv2.blur(frame, kernel)
        return MyFrame(frame)
