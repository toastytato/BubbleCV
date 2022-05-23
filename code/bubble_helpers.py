from inspect import FrameInfo
import math
import os

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial as spatial

from misc_methods import MyFrame

### Processing ###
# - Functions for processing the bubbles


class Bubble:

    REMOVED = 0
    AUTO = 1
    SELECTED = 2
    MANUAL = 3
    MANUAL_SEL = 4

    id_cnt = 0

    def __init__(self, x, y, r, frame):
        self.id = Bubble.id_cnt
        Bubble.id_cnt += 1
        self.state = self.AUTO

        self.x_list = [x]
        self.y_list = [y]
        self.r_list = [r]
        self.frame_list = [frame]
        self.neighbors = []
        self.distances = []
        self.angles = []

    def __repr__(self) -> str:
        return f'Bubble{self.id}({self.state}) at x:{self.x}, y:{self.y}\n'

    @classmethod
    def reset_id(cls):
        cls.id_cnt = 0

    def update(self, x, y, r, frame):
        # frame hasn't changed
        # but parameters have
        if frame == self.frame_list[-1]:
            self.x_list[-1] = x
            self.y_list[-1] = y
            self.r_list[-1] = r
        else:
            self.x_list.append(x)
            self.y_list.append(y)
            self.r_list.append(r)
            self.frame_list.append(frame)

    @property
    def x(self):
        return self.x_list[-1]

    @property
    def y(self):
        return self.y_list[-1]

    @property
    def r(self):
        return self.r_list[-1]

    # integer radius
    @property
    def ir(self):
        return int(self.r)

    @property
    def frame(self):
        return self.frame_list[-1]

    # returns positions as integers
    @property
    def ipos(self):
        return (int(self.x), int(self.y))

    @property
    def pos(self):
        return (self.x, self.y)

    @property
    def diameter(self):
        return self.r * 2

    @diameter.setter
    def diameter(self, d):
        self.r = d / 2


# Collection of Bubbles
class BubblesGroup:
    # allow manual centers and tree from trees created elsewhere
    def __init__(self, bubbles):
        # 'all': history of all bubbles
        # 'curr': current bubbles that are visible
        self.bubbles = {'all': bubbles, 'curr': bubbles}
        self.kd_tree = {'all': None, 'curr': None}
        self.set_curr_bubbles(bubbles)

        self.manual_bubble = None

    def set_curr_bubbles(self, bubbles):
        self.bubbles['curr'] = bubbles
        if len(bubbles) > 0:
            self.kd_tree['curr'] = spatial.KDTree([b.pos for b in bubbles])
        else:
            self.kd_tree['curr'] = None

    # set empty bubble + kd tree
    def clear(self):
        Bubble.reset_id()
        self.bubbles['all'] = self.bubbles['curr'] = []
        self.kd_tree['curr'] = None

    def export_img_of_centers(self):
        export_frame = self.frame.copy()
        for b in self.bubbles['all']:
            if b.state == Bubble.SELECTED or b.state == Bubble.MANUAL:
                export_frame = cv2.circle(export_frame, b.ipos, 3, (0, 0, 255),
                                          -1)

    def create_manual_bubble(self, pos, r, frame_idx, state):
        self.manual_bubble = Bubble(pos[0], pos[1], r, frame_idx)
        self.manual_bubble.state = state

        self.bubbles['all'].append(self.manual_bubble)
        self.bubbles['curr'].append(self.manual_bubble)
        self.set_curr_bubbles(self.bubbles['curr'])

    # get the bubble that contains the point within its radius
    def get_bubble_containing_pt(self, point, type='curr'):
        sel_bubble = None
        # check all the bubbles to see if cursor is inside
        for b in self.bubbles[type]:
            # if cursor within the bubble
            if math.dist(point, b.pos) < b.r:
                if sel_bubble is None:
                    sel_bubble = b
                # if cursor within multiple bubbles, select the closer one
                else:
                    if (math.dist(point, b.pos) < math.dist(
                            point, sel_bubble.pos)):
                        sel_bubble = b
        return sel_bubble

    def get_bubbles_of_state(self, state, type='all'):
        print(self.bubbles['all'])
        return [b for b in self.bubbles['all'] if b.state == state]

    def remove_bubble(self, bubble):
        self.bubbles['all'].remove(bubble)
        self.bubbles['curr'].remove(bubble)

    def toggle_state_bubble_containing_pt(self, point):
        b = self.get_bubble_containing_pt(point, 'curr')
        if b is not None:
            if b.state == Bubble.AUTO or b.state == Bubble.MANUAL:
                b.state = Bubble.SELECTED
            elif b.state == Bubble.SELECTED:
                b.state = Bubble.AUTO
            return True
        return False

    def set_state_bubble_containing_pt(self, point, state):
        b = self.get_bubble_containing_pt(point, 'curr')
        if b is not None:
            b.state = state
            return True
        return False

    def set_state_bubble_in_roi(self, roi, state):
        if len(roi) > 0:
            for b in self.bubbles['curr']:
                if (b.x - b.r > roi[0] and b.x + b.r < roi[0] + roi[2]
                        and b.y - b.r > roi[1]
                        and b.y + b.r < roi[1] + roi[3]):
                    b.state = state

    def correlate_other_ids_to_self(self, other_group, type='curr'):
        # take the smaller bunch
        self.kd_tree[type] = spatial.KDTree(
            [b.pos for b in self.bubbles[type]])

        for self_idx, self_b in enumerate(self.bubbles[type]):
            dist, nn_other_b = other_group.get_nearest_bubble_to_pt(self_b.pos)
            dist, nn_self_b_idx = self.kd_tree[type].query(nn_other_b.pos, 1)
            print('other:', nn_other_b, 'self:', nn_self_b_idx, self_b)
            if self_idx == nn_self_b_idx:
                nn_other_b.id = self_b.id
                nn_other_b.state = self_b.state

        # if len(other_bubbles) < len(self.all_bubbles):
        #     # assign other bubble's id to curr bubbles
        #     for other_b in other_bubbles:
        #         dist, self_nn_b = self.get_nearest_bubble_to_pt(other_b.pos)
        #         self_nn_b.id = other_b.id
        #         self_nn_b.state = other_b.state
        # else:
        #     # receive other bubble's id to curr bubbles
        #     for self_b in self.bubbles:
        #         dist, other_nn_b = other_group.get_nearest_bubble_to_pt(
        #             self_b.pos)
        #         self_b.id = other_nn_b.id
        #         self_b.state = other_nn_b.state

    # if len(self.bubbles) > len(new_marker_pts):
    # self.bubbles will have new bubbles added corresponding to new markers
    # if len(self.bubbles) < len(new_marker_pts):
    # self.bubbles will retain unassociated bubbles --> those bubbles no longer
    # have a corresponding associate in the current frame
    def update_bubbles_to_new_markers(self,
                                      new_markers,
                                      frame_idx,
                                      max_dist=200):
        # if initializing:
        if self.kd_tree['curr'] is None or len(new_markers) == 0:
            new_bubbles = [
                Bubble(m[0], m[1], m[2], frame_idx) for m in new_markers
            ]
            # no bubbles were set before
            # create new bubble set from markers (full init)
            if len(self.bubbles['all']) == 0:
                self.bubbles['all'] = new_bubbles
            self.set_curr_bubbles(new_bubbles)
            return

        # ------------------------------------------
        # else if updating:
        # find where the bubble most likely came from
        # for all of the new markers
        # see which of the previous bubbles is closest to that marker
        # if none is associated, create a new bubble

        # temporary tree for new markers
        new_markers_tree = spatial.KDTree([m[:2] for m in new_markers])

        # need a new bubble list so prev bubble list is not overwritten
        new_bubbles = []
        for new_idx, m in enumerate(new_markers):
            # find markers from prev frame closest to markers in curr frame
            dist1, nn_old_bubble = self.get_nearest_bubble_to_pt(m[:2])
            # find markers from curr frame closest to markers in prev frame
            dist2, nn_new_idx = new_markers_tree.query(nn_old_bubble.pos, 1)
            # check if the nearest neighbor to the nearest neighbor gives the same object
            # indicates that they are both the closest and thus associate the two
            if new_idx == nn_new_idx and dist1 < max_dist:
                # gets the bubble object closest to curr point
                nn_old_bubble.update(x=m[0], y=m[1], r=m[2], frame=frame_idx)
                new_bubbles.append(nn_old_bubble)
            # the prev and curr nearest neighbors do not agree
            else:
                b = Bubble(m[0], m[1], m[2], frame_idx)
                new_bubbles.append(b)
                self.bubbles['all'].append(b)
        # finished associating to previous frame's bubbles
        # we can update that to the current bubble's kd tree
        self.kd_tree['curr'] = new_markers_tree
        self.bubbles['curr'] = new_bubbles
        # marker_kd_tree become the new bubble_kd_tree

    # pos: (x, y) coordinate
    def get_nearest_bubble_to_pt(self, point):
        # print('pos:', pos)
        d, i = self.kd_tree['curr'].query(point, k=1)
        return d, self.bubbles['curr'][i]

    def get_self_neighbors(self, num_neighbors):
        # add 1 b/c it'll count the current bubble as neighbor
        dist_list, neighbor_idx_list = self.kd_tree['curr'].query(
            self.bubbles['curr'], k=num_neighbors + 1)
        return dist_list, neighbor_idx_list


class BubbleSubAnalysis(BubblesGroup):

    def __init__(self, bubbles, frame_idx, filters):
        # check if is a list of bubbles
        super().__init__(bubbles)
        self.frame_idx = frame_idx
        self.filters = filters


# receives a frame with each contour labeled
# draws a circle around each contour and returns the list of bubbles
# get kd tree of previous frame for associating IDs with temporal coherence
def get_markers_from_label(labeled_frame,
                           min_area=1,
                           fit_circle='area',
                           timeout_cnt=500):
    new_markers_pts = []

    # print('mf len', len(np.unique(markers_frame)))

    # cycles through each unique blob one by one
    # compared to threshold which gets contours from all white regions

    for label in np.unique(labeled_frame):
        if label >= timeout_cnt:
            break
        # if the label is zero, we are examining the 'background'
        # if label is -1, it is the border and we don't need to label it
        # so simply ignore it
        if label == 1 or label == -1:
            continue
        # otherwise, allocate memory
        # for the label region and draw
        # it on the mask
        mask = np.zeros(labeled_frame.shape, dtype='uint8')
        mask[labeled_frame == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > min_area:
            # draw a circle enclosing the object
            # keep r if using min enclosing circle radius
            ((x, y), r) = cv2.minEnclosingCircle(c)
            # get center via Center of Mass
            # M_1 = cv2.moments(c)
            # if M_1['m00'] == 0:
            #     M_1['m00', 'm01'] = 1
            # x = int(M_1['m10'] / M_1['m00'])
            # y = int(M_1['m01'] / M_1['m00'])
            if fit_circle == 'area':
                # 1.5 because the circle looks small
                r = math.sqrt(1.5 * area / math.pi)
            elif fit_circle == 'perimeter':
                r = cv2.arcLength(c, True) / (2 * math.pi)

            new_markers_pts.append((x, y, r))

    return new_markers_pts
    # for each marker in curr frame, see which bubble from the previous frame is nearest
    # for that prev-frame-bubble, see its nearest neighbor in the curr frame is also the same marker
    # if they're the same, we can correlate their identity
    # if not, this bubble is new and needs to be created

    # fill in bubbles after Bubbles have been found


# takes in binary grayscale image
def get_bubbles_from_threshold(frame, min_area=1):
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    # frame = MyFrame(frame)
    # gray = frame.cvt_color('gray')
    gray = frame
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    bubbles = []
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        for i, c in enumerate(cnts):
            if cv2.contourArea(c) > min_area:
                ((x, y), r) = cv2.minEnclosingCircle(c)
                # M = cv2.moments(c)
                bubbles.append(Bubble(x, y, r, -1))
    # x, y, diameter, top, bottom, left, right

    return bubbles


def get_bounds(bubbles, scale_x, scale_y, offset_x, offset_y):
    centers = [(b.x, b.y) for b in bubbles]
    lower_bound = np.min(centers,
                         axis=0) - (scale_x - offset_x, scale_y - offset_y)
    upper_bound = np.max(centers,
                         axis=0) + (scale_x + offset_x, scale_y + offset_y)
    in_id = 0
    out_id = -1
    for b in bubbles:
        if (
                # b.x - b.r > lower_bound[0]
                # and b.x + b.r < upper_bound[0]
                # and b.y - b.r > lower_bound[1]
                # and b.y + b.r < upper_bound[1]
                b.x >= lower_bound[0] and b.x <= upper_bound[0]
                and b.y >= lower_bound[1] and b.y <= upper_bound[1]):
            b.id = in_id
            in_id += 1
        else:
            b.id = out_id
            out_id -= 1

    return lower_bound, upper_bound


# REMOVED bubble will still show up here as a neighbor
def set_neighbors(kd_tree, num_neighbors):
    if (kd_tree is None or not kd_tree.is_ready()
            or len(kd_tree.bubbles) <= num_neighbors):
        return

    bubbles = kd_tree.bubbles
    dist_list, neighbor_idx_list = kd_tree.get_self_neighbors(num_neighbors)
    for neighbor_set, dist_set in zip(neighbor_idx_list, dist_list):
        # center bubble idx (self), which is also the current index of the list
        center_idx = neighbor_set[0]
        x = bubbles[center_idx].x
        y = bubbles[center_idx].y

        if bubbles[center_idx].id >= 0:
            neighbors = []
            distances = []
            angles = []
            for n, d in zip(neighbor_set[1:], dist_set[1:]):
                neighbors.append(bubbles[n])
                distances.append(d)
                angle = math.degrees(
                    math.atan2(bubbles[n].y - y, bubbles[n].x - x))
                angles.append(angle)

            bubbles[center_idx].neighbors = neighbors
            bubbles[center_idx].distances = distances
            bubbles[center_idx].angles = angles


def draw_annotations(frame, bubbles, min, max, highlight_idx, circum_color,
                     center_color, neighbor_color):
    # draw bounds
    cv2.rectangle(frame, (int(min[0]), int(min[1])),
                  (int(max[0]), int(max[1])), (100, 24, 24), 2)

    sel_bubble = None
    for b in bubbles:
        if b.id >= 0:
            if b.id == highlight_idx:
                sel_bubble = b
                continue
            # highlight all bubbles within bounds
            rgba = circum_color.getRgb()
            bgr = (rgba[2], rgba[1], rgba[0])
            cv2.circle(frame, b.ipos, int(b.r), bgr, 2)

    # highlight selected and neighbors
    if sel_bubble is not None:
        rgba = center_color.getRgb()
        bgr = (rgba[2], rgba[1], rgba[0])
        cv2.circle(
            frame,
            (int(sel_bubble.x), int(sel_bubble.y)),
            int(sel_bubble.r),
            bgr,
            2,
        )

        for n in sel_bubble.neighbors:
            rgba = neighbor_color.getRgb()
            bgr = (rgba[2], rgba[1], rgba[0])
            cv2.circle(frame, n.ipos, int(n.r), bgr, 2)

    return frame


def export_imgs(frame, bubbles, lasers, deposits):
    b_frame = frame.cvt_color('bgr').copy()
    l_frame = frame.cvt_color('bgr').copy()
    d_frame = frame.cvt_color('bgr').copy()

    for b in lasers:
        l_frame = cv2.circle(l_frame, b.ipos, 3, (0, 0, 255), -1)
    for b in deposits:
        d_frame = cv2.circle(d_frame, b.ipos, 3, (255, 0, 0), -1)

    cv2.imwrite('export/deposits_pos.png', d_frame)
    cv2.imwrite('export/lasers_pos.png', l_frame)


def export_all_bubbles_excel(bubbles,
                             roi,
                             framerate,
                             conversion,
                             url,
                             lasers=None,
                             deposits=None):
    print('Exporting CSV')
    # for filtering out bubbles generated from noise
    with pd.ExcelWriter('export/bubble_data.xlsx') as w:
        info_df = pd.DataFrame()
        info_df['roi (x, y, w, h) px'] = [roi]
        info_df['framerate'] = [framerate]
        info_df['um/pixel'] = [conversion]
        info_df['file'] = [url]
        if len(lasers) > 0:
            info_df['laser frame'] = lasers[0].frame
        if len(deposits) > 0:
            info_df['deposits frame'] = deposits[0].frame
        info_df.to_excel(w, sheet_name='INFO', index=False)

        # export the position of the lasers in a single sheet
        if lasers is not None:
            lasers_df = pd.DataFrame()
            lasers_df['frame'] = [b.frame for b in lasers]
            lasers_df['id'] = [b.id for b in lasers]
            lasers_df['x (um)'] = [b.x * conversion for b in lasers]
            lasers_df['y (um)'] = [b.y * conversion for b in lasers]
            # lasers_df['time (s)'] = [
            #     round(b.frame / framerate, 2) for b in lasers
            # ]
            lasers_df.to_excel(w, sheet_name='LASERS', index=False)

        # export the position of the deposits in a single sheet
        if deposits is not None:
            deposits_df = pd.DataFrame()
            deposits_df['frame'] = [b.frame for b in deposits]
            deposits_df['id'] = [b.id for b in deposits]
            deposits_df['x (um)'] = [b.x * conversion for b in deposits]
            deposits_df['y (um)'] = [b.y * conversion for b in deposits]
            # lasers_df['time (s)'] = [
            #     round(b.frame / framerate, 2) for b in deposits
            # ]
            deposits_df.to_excel(w, sheet_name='DEPOSITS', index=False)

        # export the time series data for each bubble
        # eqch bubble gets its own sheet
        # each sheet contains the bubble movement over time
        for b in bubbles:
            # ignore bubbles who existed less than min frames
            # they are most likely noise
            # if (len(b.frame_list) > min_existence_time
            print('Reached', f'B{b.id}')
            df = get_dataframe_from_bubble(b, framerate, 'um', conversion)
            df.to_excel(w, sheet_name=f'B{b.id}', index=False)
    print('Done!')


def get_dataframe_from_bubble(bubble, framerate, units='px', conversion=1):
    df = pd.DataFrame()

    df['frame index'] = bubble.frame_list
    df['time (s)'] = [round(f / framerate, 2) for f in bubble.frame_list]
    df[f'x ({units})'] = [x * conversion for x in bubble.x_list]
    df[f'y ({units})'] = [y * conversion for y in bubble.y_list]
    df[f'r ({units})'] = [r * conversion for r in bubble.r_list]
    df[f'volume ({units}^3)'] = [(4 / 3) * math.pi * r**3
                                 for r in df[f'r ({units})']]
    # df['um/pixel'] = conversion
    return df


def export_csv(bubbles, conversion, url):

    df = pd.DataFrame()
    df['id'] = [b.id for b in bubbles if b.id >= 0]
    df['x'] = [b.x for b in bubbles if b.id >= 0]
    df['y'] = [b.y for b in bubbles if b.id >= 0]
    df['diameter'] = [b.diameter for b in bubbles if b.id >= 0]
    df['neighbors'] = [[n.id for n in b.neighbors] for b in bubbles
                       if b.id >= 0]
    df['distances'] = [np.around(b.distances, 2) for b in bubbles if b.id >= 0]
    df['angles'] = [np.around(b.angles, 2) for b in bubbles if b.id >= 0]
    df['units'] = 'pixels'
    df['um/px'] = conversion

    name = os.path.splitext(url)[0]
    name = os.path.basename(name)

    path = get_save_dir('analysis', url) + '/datapoints.csv'
    df.to_csv(path, index=False)
    print('exported')


def export_boxplots(bubbles, num_neighbors, conversion, url):

    #
    fig, ax = plt.subplots()

    diam_vs_dist = {}

    for b in bubbles:
        if b.id >= 0:
            diam = np.rint(b.diameter * conversion)
            if diam not in diam_vs_dist:
                diam_vs_dist[diam] = [d * conversion for d in b.distances]
            else:
                diam_vs_dist[diam].extend(
                    [d * conversion for d in b.distances])

    sorted_diam_vs_dist = sorted(diam_vs_dist.items())

    ax.boxplot([b[1] for b in sorted_diam_vs_dist])
    ax.set_xticklabels([b[0] for b in sorted_diam_vs_dist])

    ax.set_ylabel(f'Distances to {num_neighbors} nearest neighbors (um)')
    ax.set_xlabel('Nearest Integer Diameter (um)')

    # plt.show()
    plt.tight_layout()
    # plt.ioff()

    path = get_save_dir('analysis', url) + '/diam_vs_dist_box.png'
    plt.savefig(path)
    plt.show()


def export_scatter(bubbles, num_neighbors, conversion, url):
    diam = [b.diameter * conversion for b in bubbles if b.id >= 0]
    distances = [np.around(b.distances, 2) for b in bubbles if b.id >= 0]
    distances = [d * conversion for d in distances]

    fig, ax = plt.subplots()

    for d, l in zip(diam, distances):
        ax.scatter([d] * len(l), l)

    ax.set_ylabel(f'Distances to {num_neighbors} nearest neighbors (um)')
    ax.set_xlabel('Diameter (um)')

    plt.tight_layout()
    # plt.ioff()

    path = get_save_dir('analysis', url) + '/dist_vs_diam_scatter.png'

    plt.savefig(path)
    plt.show()


def export_dist_histogram(bubbles, num_neighbors, conversion, url):
    fig, ax = plt.subplots()

    dist = []
    for b in bubbles:
        if b.id >= 0:
            dist = np.append(dist, [d * conversion for d in b.distances])

    bins = np.arange(0, np.amax(dist), 1)
    ax.hist(dist, bins)

    ax.set_ylabel('Number of distances')
    ax.set_xlabel('Nearest Integer Distances (um)')

    path = get_save_dir('analysis', url) + '/dist_histogram.png'
    plt.savefig(path)
    plt.show()


def export_diam_histogram(bubbles, num_neighbors, conversion, url):
    fig, ax = plt.subplots()

    diam = []
    for b in bubbles:
        if b.id >= 0:
            diam = np.append(diam, b.diameter)

    bins = np.arange(0, np.amax(diam), 1)
    ax.hist(diam, bins)

    ax.set_ylabel('Number of diameters')
    ax.set_xlabel('Nearest Integer Diameter (um)')

    # if not os.path.exists('analysis/histograms'):
    #     os.makedirs('analysis/histograms')
    # name = os.path.splitext(url)[0]
    # name = os.path.basename(name)
    path = get_save_dir('analysis', url) + '/diam_histogram.png'
    plt.savefig(path)
    plt.show()


def get_save_dir(main_path, url):
    name = os.path.splitext(url)[0]
    name = os.path.basename(name)
    path = f'{main_path}/{name}'
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# %%
