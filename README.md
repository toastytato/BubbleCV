# Bubble Analysis OpenCV Interface

A multhreaded PyQt UI built to allow for easy implementation of OpenCV and other CV libraries by displaying filters/processes visually with easy tuning of parameters. Built as part of research for Dr.Fan at University of Texas.

## Description

The processing of frames works in the following order:

1. **Filtering**: performing image transformations and manipulations to obtain desired objects of interest
2. **Processing**: performing image operations to obtain data after it has been filtered
3. **Annotations**: drawing overlays onto frame to help visualize what is being done by the operations

The purpose of each file is as follows:

- `main.py`: The UI container for displaying views and handling signals from display elements. It also holds the video thread for performing the image operations.
- `main_params.py`: The main parameter tree where child parameter objects called and stored.
- `filter_params.py`: Holds the parameters for each of the filter objects as well as the filter operations.
- `processing_params.py`: Holds the parameters for each of the processing objects as well as the processing operations.
- `filter.py`: Holds the all the image filtering operations. 
- `bubble_process.py`: Holds the image operations for the bubble processing object.

## Installation:

Install Python 3.8 or higher

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install necessary packages once this repository has been cloned.

```bash
pip install git+git://github.com/pyqtgraph/pyqtgraph.git@master
pip install PyQt5
pip install pandas
pip install imutils
pip install opencv-python
pip install scipy
pip install dataclasses
pip install matplotlib
```

(Note: `pip install pyqtgraph` will NOT download the proper pyqtgraph version to run this software. Make sure to follow installation using `git` as shown above. This will get the latest version of pyqtgraph from the source.)

## Usage

### General
1. Click the file select button to open a file explorer and choose the desired image to analyze
2. Click `Select ROI` to choose the region of interest. A new window will pop up, and use the mouse and cursor to select the region. Press *Enter* or *Space* to confirm, or click *c* or *esc* to cancel. 
3. To isolate desired elements of interest, click `Add` dropdown in the Filter parameter group to add a filter. Description of filters below. Stack filters as necessary
4. Right click on the filter names to move them up or down (the filter operation works top down, so order matters) or to delete any filter.
5. To analyze, click on the `Add` dropdown in the Analyze parameter group to add a processing operation. The operations here does not stack, meaning they will all work off of the last filtered image. However, their image annoatations will stack on top of each other. Description or processes below. 

### Filters
- `Threshold`: Converts colors above threshold to white and black otherwise. 
    - `thresh`: Normal threshold
    - `inv thresh`: Inverts the result of the threshold
    - `otsu`: Automatic threshold
- `Watershed`: In development, could be used to solve merging issue with bubbles being too close to each other

### Processes
- `Bubble`: Analyzes thresholded images to identify contours and their respective distances + diameters.
    - `Min Size`: The minimum area of contour to identify. Ignores all contours with areas smaller then this threshold
    - `Num Neighbors`: Number of neighbors to identify to record relative distances
    - `Bounds Offset X/Y`: Shifts the automatically generated bounds. Bubble centers outside of bounds are ignored.
    - `Bounds Scale X/Y`: Scales the automatically generated bounds. Bubble centers outside of bounds are ignored.
    - `Conversion`: Shows the unit conversion from pixels to um when exporting data (not modifiable)
    - `Export Distances`: Exports a csv file with the data of all bubbles that were of interest. This includes their id, x y positions, diameter, neighbor ids as a list, distances to neighbors as a list, angle to neighbors as a list, units, and conversion from px to um. Note: negative ID means that the bubbles were outside of the bounds but still counted as neighbors (for bubbles on the edge of bounds this will happen)
    - `Export Graphs`: Exports all graphs related to bubbles of interest. This includes a scatterplot of Diameter vs Distance to n nearest neighbors, a boxplot of Nearest Integer Diameter vs Distance to n nearest neighbors, a histogram of Nearest Integer Distances (um), and a histogram of Nearest Integer Diameters (um). 
    - `Bubble Highlight`: Highlights a bubble based on their ID 
    - `Center Color`: Color of main highlighted bubble
    - `Circumference Color`: Color of all bubbles of interest
    - `Neighbor Color`: Color of neighbors




## Bugs
- ~~On new file select the display won't show it filtered. Turn the filters on and off to fix.~~
- When reordering filters, sometimes it crashes because of some loss of reference to the PyQt object. So try to avoid reordering if possible. Delete then add would be safer.
- Scrolling on top of the dropdown menu will add new filters. Try to avoid that if not intended?


## Contributing
Yes

## References

[PyQtGraph Documentation](https://pyqtgraph.readthedocs.io/en/latest/)  
[PyQtGraph Website](https://www.pyqtgraph.org/)  
https://pretagteam.com/question/find-contours-after-watershed-opencv

Note: Documentation doesn't cover everything, I had to look in the codebase to understand how to utilize certain functions
