
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/maxcrous/multiview_notebooks/main)

# Multiview notebooks
A collection of educational notebooks on multi-view geometry and computer vision.
Subjects covered in these notebooks include

- Camera calibration
- Perspective projection
- 3D point triangulation
- Quaternions as 3D pose representation
- Perspective-n-point (PnP) algorithm
- Levenberg–Marquardt optimization 
- Epipolar geometry
- Relative poses from stereo views
- Bundle adjustment
- Structure from motion

# How to run 
The notebooks can be run in the browser by clicking the binder badge 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/maxcrous/multiview_notebooks/main).
If one is interested in running the notebooks locally, I highly recommend using Docker as there is a dependency on g2opy and ipyvolume, which are challenging to install. 

```
# Builds the environment 
docker build -t multiview_notebooks .

# Start a jupyter lab which can be opened in the browser
docker run -it --rm -p 8888:8888 my-image jupyter-lab --ip=0.0.0.0 --port=8888
```
After starting the jupyter lab, the notebooks can be found in the home directory.   
For the source of the Dockerfile, see [this repository](https://github.com/maxcrous/ipyvolume_g2opy_notebooks)
