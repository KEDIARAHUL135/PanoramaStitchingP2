# PanoramaStitchingP2

This project is created over the project [Panorama Stitching](https://github.com/KEDIARAHUL135/PanoramaStitching.git). 
This project uses the concept of the previous project and applies it to stitch multiple images together to create a wide-view panorama.
Code is available in Python and C++.

You can find complete explaination of the logic and documentation of the code [here](https://www.scribd.com/document/510892625/Panorama-Stitching-P2).


### Installation

Clone this repository and follow the steps below to run the code.

##### Python
* Install the following dependencies.
    * `opencv-python`
    * `numpy`
* Set the path of the folder containing input images in the file `main.py` at lines 239.
* Run the code using terminal
    * Navigate to the cwd.
    * Run: `$ python main.py`
* The output will be stored in the cwd by the name of "Stitched_Panorama.png".
    
#### C++
* Make sure you have OpenCV binaries installed.
* Set the path of the folder containing input images in the file `main.cpp` at lines 386.
* Run the file `main.cpp`. For CMake users, `CMakeLists.txt` is also created.
