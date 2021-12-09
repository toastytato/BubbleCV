import cv2
import numpy as np
from pyqtgraph.parametertree import registerParameterType
'''
Methods shared between multiple files
'''

# frame = (cv2.imread(), 'gray')


# register my custom param types on instantiation of the class
def register_my_param(param_obj):
    print("Registering:", param_obj.cls_type)
    registerParameterType(param_obj.cls_type, param_obj)
    return param_obj


# wrapper around frame to give it color attribute
class MyFrame(np.ndarray):
    # cls should be of MyFrame type
    def __new__(cls, input_array, colorspace=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.colorspace = colorspace.lower()
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
        end_clr = end_clr.lower()
        print("From:", self.colorspace, "To:", end_clr)
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
        return MyFrame(cv2.cvtColor(self, color_map[self.colorspace][end_clr]),
                       end_clr)