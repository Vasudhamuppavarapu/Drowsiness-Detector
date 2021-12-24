# Drowsiness-Detector
Pre-requisites (Install the below libraries USING pip install command) 

1. Python 3
2. opencv (tested with 3.4) 
3. dlib	(tested with 19.18.0)
4. imutils (tested with 0.5.3)
5. scipy
6. numpy
7. argparse

NOTE: dlib might not be installed, to ensure it is successfully installed;
1. Download and install Visual Studio Community edition
2. Install cmake and wheel using the below commands;
pip install cmake
pip install wheel




How to Run 
There are two ways to run the code;
1. Is using Pycharm
Right click on the .py file and open with Pycharm
Change the directory to open in project
The pycharm dependancies will appear so install all
Edit and configure interpreter
If using external camera, dont edit, if using internal camera, check your camera value and change in the code from zero
Click run

2. Using command

Python3.9 Fatigue_Detector.py -- webcam 0	  // webcam 0 is for internal camera only and if you are using internal camera, check for its webcam accordingly.


Configurations
You can change the below thresholds accordingly.

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 10`	


