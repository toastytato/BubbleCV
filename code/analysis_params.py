from PyQt5.QtCore import QObject, Qt, pyqtSignal
from pyqtgraph.parametertree import Parameter
import pyqtgraph.parametertree.parameterTypes as ptypes
import cv2
import numpy as np
'''
Analysis Params: Any new analysis methods could be easily implemented here
with its associating parameters. 
def analyze(frame): called in the video/image processing thread and the frame is passed in for processing
- does not need to return anything. state data of the operation is retained here.
def annotate(frame): also in the video thread, called on last after all analysis to draw on annotations. 
- returns annotated
'''


# boiler plate code for analysis params
class Analysis(Parameter):
    '''
    Variables for access:
    self.orig_frame: the uncropped, unfiltered input frame
    self.cursor_pos: x, y coordinate of mouse position
    '''

    # params: analyze: bool, annotate: bool
    # tells the main controller whether or not
    # to update analysis/annotations
    request_analysis_update = pyqtSignal()
    request_annotate_update = pyqtSignal()
    # true for play, false for pause
    request_resume = pyqtSignal(bool)
    # set curr frame
    request_set_frame_idx = pyqtSignal(int)
    # update video url
    request_url_update = pyqtSignal(str)

    def __init__(self, **opts):
        opts['removable'] = True
        if 'name' not in opts:
            opts['name'] = 'DefaultAnalysis'

        if 'children' not in opts:
            opts['children'] = []

        all_children_names = [c['name'] for c in opts['children']]

        self.file_param = ptypes.FileParameter(name="File Select",
                                               value=opts.get('url', ""))
        self.frame_idx_param = ptypes.SimpleParameter(name='Curr Frame',
                                                      type='int',
                                                      value=opts.get(
                                                          'curr_frame_idx', 0))
        self.playback_param = ptypes.ActionParameter(title='Play',
                                                     name="Playback",
                                                     type="action")
        self.roi_param = ptypes.ActionParameter(name='Select ROI')
        self.roi_readout_param = ptypes.SimpleParameter(name='Input ROI',
                                                        type='str')
        self.cursor_info_param = ptypes.SimpleParameter(name='Cursor Info',
                                                        type='str',
                                                        readonly=True,
                                                        value='')
        self.settings = Parameter.create(
            name='Settings',
            type="group",
            children=[
                self.file_param, self.frame_idx_param, self.playback_param,
                self.roi_param, self.roi_readout_param, self.cursor_info_param
            ])
        opts['children'].insert(0, self.settings)

        # if 'overlay' not in all_children_names:
        #     opts['children'].append({
        #         'title':
        #         'Overlay',
        #         'name':
        #         'overlay',
        #         'type':
        #         'group',
        #         'children': [{
        #             'name': 'Toggle',
        #             'type': 'bool',
        #             'value': True
        #         }]
        #     })
        print(opts)
        super().__init__(**opts)

        # print(self.opts)

        # for c in self.opts['children']:
        #     print('c:', c)
        #     if c['type'] != 'group':
        #         setattr(self, f"get_{c['name']}", lambda: c.value())
        # self.sigRemoved.connect(self.on_removed)'
        self.file_param.sigValueChanged.connect(
            lambda x: self.request_url_update.emit(x.value()))

        self.sigTreeStateChanged.connect(self.on_param_change)

        self.roi_param.sigActivated.connect(self.set_roi)
        self.roi_readout_param.sigValueChanged.connect(self.input_roi)
        self.frame_idx_param.sigValueChanged.connect(
            self.send_frame_idx_request)
        self.playback_param.sigActivated.connect(self.toggle_playback)

        # manual sel states:
        self.opts['roi'] = None
        self.orig_frame = np.array([])
        self.cursor_pos = [0, 0]
        self.is_playing = False
        self.has_ended = False

    # put algorithmic operations here
    # called on parameter updates
    def analyze(self, frame):
        self.orig_frame = frame

    # put lighter operations
    # in here which will be called on every update
    def annotate(self, frame):
        return frame

    # placeholder, should overwrite when subclassing
    def on_param_change(self, parameter, changes):
        for param, change, data in changes:
            print(f'{param.name()=}, {change=}, {data=}')

    def send_frame_idx_request(self, param):
        self.request_set_frame_idx.emit(int(param.value()))

    @property
    def curr_frame_idx(self):
        return self.frame_idx_param.value()

    # will tell video thread to update to current frame
    @curr_frame_idx.setter
    def curr_frame_idx(self, idx):
        self.frame_idx_param.setValue(idx)

    # will NOT tell video thread to update to current frame
    # only updates view
    # prevent recursive call to itself due to signal being triggered
    def set_curr_frame_idx_no_emit(self, idx):
        self.frame_idx_param.sigValueChanged.disconnect(
            self.send_frame_idx_request)
        self.frame_idx_param.setValue(idx)
        self.frame_idx_param.sigValueChanged.connect(
            self.send_frame_idx_request)

    @property
    def is_playing(self):
        # shows pause while playing and vice versa
        return (self.playback_param.title() == 'Pause')

    # True: is playing
    @is_playing.setter
    def is_playing(self, resume):
        if resume:
            self.playback_param.setOpts(title='Pause')
            # self.child('analysis_group', 'watershed', 'Toggle').setValue(True)
        else:
            self.playback_param.setOpts(title='Play')

    def video_ended(self, state):
        self.has_ended = state
        self.is_playing = False

    def toggle_playback(self):
        self.is_playing = not self.is_playing

    def crop_to_roi(self, frame):
        if self.opts['roi'] is not None:
            # print('pre:', frame.shape)
            # print('roi:', self.opts['roi'])
            frame = frame[int(self.opts['roi'][1]):int(self.opts['roi'][1] +
                                                       self.opts['roi'][3]),
                          int(self.opts['roi'][0]):int(self.opts['roi'][0] +
                                                       self.opts['roi'][2])]
            # print('post', frame.shape)
            return frame
        else:
            print(frame.shape)
            (h, w) = frame.shape[:2]
            self.opts['roi'] = [0, 0, w, h]
            return frame

    def set_roi(self):
        r = cv2.selectROI("Select ROI", self.orig_frame)
        if all(r) != 0:
            self.opts['roi'] = r
        cv2.destroyWindow("Select ROI")
        # self.child('settings',
        #            'input_roi').setValue(f"({r[0]}, {r[1]}, {r[2]}, {r[3]})")
        self.request_analysis_update.emit()

    def input_roi(self, param):
        text = param.value().strip()[1:-1]
        text = text.split(', ', 4)
        try:
            roi = [int(s) for s in text]
        except Exception:
            return
        self.opts['roi'] = roi
        self.request_analysis_update.emit()

    def set_cursor_value(self, cursor_pos, cursor_val):
        self.cursor_info_param.setValue(
            f'x:{cursor_pos[0]}, y:{cursor_pos[1]}, {cursor_val}')

    def on_mouse_click_event(self, event):
        print('click', event)
        self.request_annotate_update.emit()

    def on_mouse_move_event(self, x, y):
        self.cursor_pos = [x, y]
        self.request_annotate_update.emit()

    def on_keypress_event(self, key):
        self.request_annotate_update.emit()

    def __repr__(self):
        msg = self.opts['name'] + ' Analysis'
        for c in self.childs:
            msg += f'\n{c.name()}: {c.value()}'
        return msg
