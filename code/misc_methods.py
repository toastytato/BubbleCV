import cv2
import numpy as np
from pyqtgraph.parametertree import registerParameterType
'''
Methods shared between multiple files
'''

# frame = (cv2.imread(), 'gray')


# decorator for register my custom param types on instantiation of the class
def register_my_param(param_obj):
    print("Registering:", param_obj.cls_type)
    registerParameterType(param_obj.cls_type, param_obj)
    return param_obj


# wrapper around frame to give it color attribute
class MyFrame(np.ndarray):
    # cls should be of MyFrame type
    def __new__(cls, input_array, colorspace=None):  #, roi=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)

        # add the new attribute to the created instance
        # if wrapping np array without a colorspace param
        if colorspace is None:
            if isinstance(input_array, np.ndarray):
                # assume 2 dimensional vector as gray
                if input_array.ndim == 2:
                    obj.colorspace = 'gray'
                # assume 3 dimensional vector as bgr
                elif input_array.ndim == 3:
                    obj.colorspace = 'bgr'
            elif isinstance(input_array, MyFrame):
                obj.colorspace = input_array.colorspace
        else:
            obj.colorspace = colorspace.lower()

        # # keep roi of the original frame
        # if roi is None:
        #     if isinstance(input_array, np.ndarray):
        #         # assume 2 dimensional vector as gray
        #         obj.roi =
        #     elif isinstance(input_array, MyFrame):
        #         obj.roi = input_array.roi
        # else:
        #     obj.roi = roi

        # Finally, we must return the newly created object:
        return obj

    # called on template creation
    # view creation
    # of instance creation
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        # returns None is 'colorspace' is not found
        self.colorspace = getattr(obj, 'colorspace', None)

    # TODO: make change self's array attribute as well
    # currently creates a new instance every time this method is called
    def cvt_color(self, end_clr):
        end_clr = end_clr.lower()  # to lowercase
        # print("From:", self.colorspace, "To:", end_clr)
        if end_clr == self.colorspace:
            return self

        color_map = {
            "bgr": {
                "gray": cv2.COLOR_BGR2GRAY,
                "hsv": cv2.COLOR_BGR2HSV,
                "rgb": cv2.COLOR_BGR2RGB
            },
            "gray": {
                "bgr": cv2.COLOR_GRAY2BGR,
                "rgb": cv2.COLOR_GRAY2RGB,
            },
            "hsv": {
                "bgr": cv2.COLOR_HSV2BGR
            }
        }
        # return a new instance of self with the right color
        try:
            m = color_map[self.colorspace][end_clr]
        except KeyError:
            print(f'No conversion from {self.colorspace} to {end_clr}')

        return MyFrame(cv2.cvtColor(np.uint8(self), m), end_clr)