import numpy as np
from moviepy.editor import VideoFileClip
import cv2
import glob

def get_edges(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    red = cv2.inRange(image, (230, 0, 0), (255, 255, 255))
    yellow = cv2.inRange(hsv, (10, 100, 100), (30, 255, 255))
    white = cv2.inRange(hsv, (100, 0, 150), (255, 85, 255))
    combined_binary = np.zeros_like(yellow)
    combined_binary[(red == 255) | (yellow == 255) | (white == 255)] = 1
    
    return combined_binary

def get_fits(warped):
    histogram = np.sum(warped[np.int(warped.shape[0]/2):,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 12
    window_height = np.int(warped.shape[0]/nwindows)
    
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50

    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

# Make a list of test images
images = glob.glob('test_images/*.jpg')

# Compute camera calibration matrix and distorition coefficients
img = cv2.imread(images[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[:2],None,None)

src = np.float32([[180, 720], [550, 470], [745, 470], [1150, 720]])
dst = np.float32([[119, 719], [127, 14], [1167, 14], [1144, 719]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

def process_image(image,mtx=mtx,dist=dist,M=M,Minv=Minv):

    undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    warped = cv2.warpPerspective(undistorted, M, undistorted.shape[:2][::-1], flags=cv2.INTER_LINEAR)
    edges = get_edges(warped)

    left_fit, right_fit = get_fits(edges)

    ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    y_eval = np.max(ploty)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(edges).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    unwarped = cv2.warpPerspective(color_warp, Minv, color_warp.shape[:2][::-1], flags=cv2.INTER_LINEAR)
    result = cv2.addWeighted(image, 1, unwarped, 0.3, 0)
      
    # CALCULATE RADIUS OF CURVATURE IN WORLD SPACE
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    curvature = (left_curverad + right_curverad) / 2.0

    # Calculate offset
    lb = left_fit[0]*result.shape[0]**2 + left_fit[1]*result.shape[0] + left_fit[2]
    rb = right_fit[0]*result.shape[0]**2 + right_fit[1]*result.shape[0] + right_fit[2]
    lanecenter = (lb+rb)/ 2
    cameracenter = result.shape[1] / 2
    pixeloffset = cameracenter-lanecenter
    worldoffset = xm_per_pix * pixeloffset

    # Combine the result with the original image    

    cv2.putText(result,'Curve Radius: %.2f m' % (curvature),(100,100),cv2.FONT_HERSHEY_COMPLEX,1.4,(0,0,255),3)
    cv2.putText(result,'Offset: %.2f m' % (worldoffset),(100,160),cv2.FONT_HERSHEY_COMPLEX,1.4,(0,0,255),3)

    return result

output = 'output_video.mp4'
clip = VideoFileClip('project_video.mp4')
process_clip = clip.fl_image(process_image)
process_clip.write_videofile(output, audio=False)