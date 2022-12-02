
### Processes

---

- Curr Frame: get bubbles
  - If prev frame exist
    - Use saved KD Tree (from prev frame) for NN ID placement
  - Else (first frame or new image/video loaded)
    - Create initial IDs
- Create new KD Tree
- Finding bubble nearest to cursor
  - use new kd tree for finding nearest bubble
- Setting neighbors
  - use new kd tree for finding neighbors

---

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
