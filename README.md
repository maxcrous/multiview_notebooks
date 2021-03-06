
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/maxcrous/multiview_notebooks/main)

## Multiview notebooks
This is a collection of educational notebooks on multi-view geometry and computer vision.
Subjects covered in these notebooks include:

- Camera calibration
- Perspective projection
- 3D point triangulation
- Quaternions as 3D pose representation
- Perspective-n-point (PnP) algorithm
- Levenberg–Marquardt optimization 
- Epipolar geometry
- Relative 2nd cam pose from stereo views w. fundamental matrix
- Relative 2nd cam pose from stereo views w. homography 
- Bundle adjustment
- Structure from motion

**Note** Notebook 5 is working but not as tidy as the rest (yet). This notebook covers the Faugeras method to infer relative pose from a homography. 

## How to run 
The notebooks can be run in the browser by clicking the binder badge 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/maxcrous/multiview_notebooks/main).
If one is interested in running the notebooks locally, I highly recommend using Docker as there is a dependency on g2opy and ipyvolume, which are challenging to install. 

```
# Builds the environment 
docker build -t multiview_notebooks .

# Start a jupyter lab which can be opened in the browser
docker run -it --rm -p 8888:8888 multiview_notebooks jupyter-lab --ip=0.0.0.0 --port=8888
```
After starting the jupyter lab, the notebooks can be found in the home directory.   
For the source of the Dockerfile, see [this repository](https://github.com/maxcrous/ipyvolume_g2opy_notebooks)

## Examples of visualizations
For more examples, see [this video on youtube](https://www.youtube.com/watch?v=mt9LZVy9G2g) 


<p align="center">
<img src="images/triangulation.gif" width="800" alt="Triangulation">
<br>
<br>
<br>
<br>
<br>
<img src="images/pnp.gif" width="800" alt="Perspective n Point">
</p>
