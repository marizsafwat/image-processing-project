# Advanced-Lane-detection-project
# Project Steps :
1] Compute the camera calibration matrix and distortion coefficients given a set of chessboard images which located in [camera_cal] .
 
2] Apply a distortion correction to images.  
 
3] Apply a perspective transform to get plan view image (prespective_forward).
 
4] first you should draw the histogram of the binary picture to detect the position of the lanes that is most likely to be found

5] using the histogram we can find the initial position of the lane

6] after finding the initial position we will apply the sliding window search which helps to detect the lane position whether it is curved or a straight line 
 
7] After getting the left lane equation and right lane equation we begin to alocate them and highlight the lane
 
8] from these polynomials we also get the radius of curvature of the left and right lanes then get their average to be the radius of curvature of the road
    and we also get the vehicle center position relative to the center of lane 

10] The input video is then sent to the process_image function which applies all the previous steps to each frame

11] The process_image functions converts the images to binary to make it easier for lane detection

12] Vehicle position and radius of curvature are then calculated by calling their functions

13] The pictures are then sent to add_text function to write the data on each frame

14] Finally the process_image function concatinates the different outputs of each stage together to show the final output
