**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This document is the writeup.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In the first code cell of the IPython notebook `advlanes.ipynb`, a regular 9x6x1 array of "object points" is constructed. This represents (x, y, z) coordinates of chessboard corners as we would expect them in an ideal chessboard with z fixed at 0. These are stored in the list `objpoints`.

I iterate through each of the chessboard calibration images and detect the chessboard corners using `cv2.findChessboardCorners()`. The detected corners are placed in the list `imgpoints`.

I then used `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  In the second code cell of the notebook under the heading "**Undistort calibration images**", the distortion correction is applied to each of the calibration images using `cv2.undistort()`.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Distortion correction is applied to each of the test images in the third code cell of the notebook under the heading "**Undistort test images**".

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

My thresholded binary image is created using a custom color detection function `get_edges`. Examples of this on the test images can be seen in the 4th code cell of the notebook under the heading "**Binary mask**".

The color detection function uses:

1. A filter to detect the R (red) channel in the input RGB space by thresholding the R channel between 230 and 255.
2. A filter to detect yellow by using a `cv2.inRange()` threshold of the HSV space between  (10, 100, 100) and (30, 255, 255).
3. A filter to detect white by using a `cv2.inRange()` threshold of the HSV space between  (100, 0, 150) and (255, 85, 255).
4. If either of the thresholds in (1), (2) or (3) are met, the combined filter passes (returns 1).

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The 5th code cell in the notebook under the heading "**Perspective Transform**" provides examples on the test images of the perspective transform I applied.

In order to obtain a birds-eye view, I chose four points on one of the example images and stored them in a `src` array. I then created a `dst` array which contained the same four points transformed to a top down view. 

The transformation matrix and it's inverse were obtained using the  `cv2.getPerspectiveTransform()` function and the warping was done using the `cv2.warpPerspective()` function, passing in the matrix.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image where the road appeared to be straight. I then examined the warped image to verify that the lines appear parallel.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The next code section under the heading "**Find lane lines and fit polynomial / draw lines on original image**" provides the code for fitting polynomials using a sliding window. The final video pipeline in the `lane_movie.py` file makes use of this sliding window approach.

The basic idea for the sliding window approach is to look at the bottom line of the binary thresholded image and use a histogram to find the areas with the highest intesity. These 2 peaks (for the left and right lanes) are then used as the start of the search which proceeds upwards by dividing the image vertically into 9 windows of fixed width. At each of the 9 steps, the windows are re-centered based on the discovered thresholded values at the step.

After the search is completed (whether bootstrapping with a histogram and sliding windows or using the last fitted line), a new line is fit using all of the discovered point. This can be accomplished using the numpy function `np.polyfit()`.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The "**Find lane lines and fit polynomial / draw lines on original image**" section also provides the code for calculating the radius of curvature and the offset with respect to center.

To find the radius of curvature, I followed the formulas derived in the lecture notes. These rely on using the fitted lines. To arrive at a single number, I average the two estimates:

```
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix +
left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])

right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix +
right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

curvature = (left_curverad + right_curverad) / 2.0
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Examples of plotting the result back on the original image can also be seen in the final code section in the notebook under the heading "**Find lane lines and fit polynomial / draw lines on original image**".

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

See the included video `output_video.mp4`.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

At first, there were parts of the sample video where the lines veered off course significantly. Shadows posed the most substantial problem, but adjacent lane lines as well as the line of the barrier in the shoulder can be detected as false positives as well.

While I experimented with smoothing over multiple video frames, ultimately this proved unnecessary as the source of my problems was an inadequate binary mask function. Focusing on isolating colors rather than looking for edges was key to making a good lane detector.

I tried the challenge video, but the result was quite poor. I believe the shadows, and perhaps the different colors in that scene compared to the ones I'm selecting for may be the biggest issues. Experimenting with "wider" masks that can find lines in a greater variety of scenes will probably help with generalizing the detector. It may then be necessary to incorporate averaging over consecutive frames.

The sliding windows algorithm could use some improvement as well. It implicitly assumes that the search will begin at the bottom of the frame. However, with the dashed lane lines sometimes the search begins higher up in the middle of the image. Searching in both directions could help here. I believe it was also mentioned in the lecture videos that convolution is an alternative approach to the sliding windows algorithm.