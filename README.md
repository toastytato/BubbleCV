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
- `filter.py`: Holds the all the image filter operations. 
- `bubble_process.py`: Holds the image operations for the bubble processing object.

## Installation:

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install necessary packages once this repository has been cloned.

```bash
# for UI
pip install git+git://github.com/pyqtgraph/pyqtgraph.git@master
pip install PyQt5

# for image processing and data export
pip install pandas
pip install imutils
pip install opencv2
pip install scipy
pip install dataclasses
pip install matplotlib
```

(Note: `pip install pyqtgraph` will NOT download the proper pyqtgraph version to run this software. Make sure to follow installation using `git` as shown above. This will get the latest version of pyqtgraph from the source.)

## References

[PyQtGraph Documentation](https://pyqtgraph.readthedocs.io/en/latest/)  
[PyQtGraph Website](https://www.pyqtgraph.org/)  
Note: Documentation doesn't cover everything, I had to look in the codebase to understand how to utilize certain functions
