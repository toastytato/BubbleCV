# Bubble Analysis OpenCV Interface

A multhreaded PyQt UI built to allow for easy implementation of OpenCV and other CV libraries by displaying filters/processes visually with easy tuning of parameters. Built as part of research for Dr.Fan at University of Texas.

## Description

The processing of frames works in the following order:

1. **Filtering**: performing image transformations and manipulations to obtain desired objects of interest
2. **Processing**: performing image operations to obtain data after it has been filtered
3. **Annotations**: drawing overlays onto frame to help visualize what is being done by the operations

The purpose of each file is as follows:

**Program**

- `main.py`: The UI container for displaying views and handling signals from display elements.
- 'video_thread.py': Image operations such as the analysis and annotations are performed on separate thread
- `main_params.py`: The main parameter tree where child parameter objects called and stored.
- `analysis_params.py`: Contains the template Analysis object for image operations.
- `filter_params.py`: Holds the parameters for each of the filter objects as well as the filter operations.
- `filter.py`: Holds the all the image filtering operations (OBSOLETE, just put the image operations inside the filter_params.py rather than creating new functions).
- `plotter.py`: Not used, ignore.

The files below are for custom analysis using the above framework.

**Tracking Analysis**

- `tracking_analysis.py`: Holds the parameters used for analyzing distinct bubble positions over time.
- `bubble_helpers.py`: Helper functions and objects used by the tracking analysis
- [Click Here](TrackingAnalysis.md) for explanation of the analysis

**Lifetime Analysis**

- `lifetime_analysis.py`: Holds the parameters used for analyzing the initial position and lifetime of bubbles close to each other
- [Click Here](LifetimeAnalysis.md) for explanation of the analysis

## Installation

Install Python 3.8 or higher

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install necessary packages once this repository has been cloned.

```bash
pip install git+git://github.com/pyqtgraph/pyqtgraph.git@master
pip install PyQt5
pip install pandas
pip install imutils
pip install opencv-python
pip install scipy
pip install scikit-image
pip install dataclasses
pip install matplotlib

```

(Note: `pip install pyqtgraph` will NOT download the proper pyqtgraph version to run this software. Make sure to follow installation using `git` as shown above. This will get the latest version of pyqtgraph from the source.)

## Usage

### General

1. Click the file select button to open a file explorer and choose the desired image to analyze
2. Click `Select ROI` to choose the region of interest. A new window will pop up, and use the mouse and cursor to select the region. Press _Enter_ or _Space_ to confirm, or click _c_ or _esc_ to cancel.
3. To isolate desired elements of interest, click `Add` dropdown in the Filter parameter group to add a filter. Description of filters below. Stack filters as necessary
4. Right click on the filter names to move them up or down (the filter operation works top down, so order matters) or to delete any filter.
5. To analyze, click on the `Add` dropdown in the Analyze parameter group to add a processing operation. The operations here does not stack, meaning they will all work off of the last filtered image. However, their image annoatations will stack on top of each other. Description or processes below.

### Filters

- `Threshold`: Converts colors above threshold to white and black otherwise.
  - `thresh`: Normal threshold
  - `inv thresh`: Inverts the result of the threshold
  - `otsu`: Automatic threshold
- `Watershed`: In development, could be used to solve merging issue with bubbles being too close to each other

## Contributing

Yes

## References

[PyQtGraph Documentation](https://pyqtgraph.readthedocs.io/en/latest/)  
[PyQtGraph Website](https://www.pyqtgraph.org/)  
<https://pretagteam.com/question/find-contours-after-watershed-opencv>

Note: Documentation doesn't cover everything, I had to look in the codebase to understand how to utilize certain functions
