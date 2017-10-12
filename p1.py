
# use `pythonw p1.py` on OSX

import sys, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import helper

DEBUG = False

CANNY_LOW = 50
CANNY_HIGH = 255

GAUSS_KERNEL = 0

# fractional width of parallelogram top
ROI_TOP_WIDTH = 0.30
# fractional height of parallelogram
ROI_TOP_HEIGHT = 0.35

# distance resolution in pixels of the Hough grid
HOUGH_RHO = 1
# angular resolution in radians of the Hough grid
HOUGH_THETA = 2 * (np.pi / 180)
# minimum number of votes (intersections in Hough grid cell)
HOUGH_THRESHOLD = 20
# minimum number of pixels making up a line
HOUGH_MIN_LINE_LEN = 10
# maximum gap in pixels between connectable line segments
HOUGH_MAX_LINE_GAP = 5

# Line Grouping

GROUP_ENABLE = True

# Grouping Parameters

# Amount of rounding for line offsets
GROUP_OFFSET = 1000.0
# Amount of rounding for slopes
GROUP_SLOPE = 0.5
# Minimum number of lines in a group
GROUP_MIN_NUM = 1
# Minimum total length of lines in a group
GROUP_MIN_SUM = 1

# Enable filtering lines by offset + slope envelope
GROUP_ITEM_ENABLE_ENVELOPE = True

# Offset + Slope envelope parameters
GROUP_RIGHT_OFFSET_MIN = -0.2
GROUP_RIGHT_OFFSET_MAX = 0.1
GROUP_RIGHT_SLOPE_MIN = 1.0
GROUP_RIGHT_SLOPE_MAX = 2.0

# Offset + Slope envelope parameters
GROUP_LEFT_OFFSET_MIN = 0.8
GROUP_LEFT_OFFSET_MAX = 1.1
GROUP_LEFT_SLOPE_MIN = -2.0
GROUP_LEFT_SLOPE_MAX = -1.2

# Line Drawing

LINE_COLOR = [0, 255, 0]
LINE_WIDTH = 4


# multiplier for lines
MERGE_ALPHA = 1.0
# multiplier for base image
MERGE_BETA = 0.4
# constant offset for base image
MERGE_LAMBDA = 0.9


def image_roi(img, top_width, top_height):        
    imshape = img.shape
    vertices = np.array([[
        (0,imshape[0]),
        (imshape[1] * (0.5 - (top_width / 2.0)), imshape[0] * (1.0 - top_height)), 
        (imshape[1] * (0.5 + (top_width / 2.0)), imshape[0] * (1.0 - top_height)), 
        (imshape[1],imshape[0])
    ]], dtype=np.int32)
    return vertices

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

def len_slope_offset(line):
    # Calculate the length, slope and offset of the line.
    # NOTE: actually using the reciprocal of the slope because most of our lines will be  close to vertical
    si = []
    for x1, y1, x2, y2 in line:
        l = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        p = np.polyfit([y1, y2], [x1, x2], 1)
        # Only process first segment of line
        return (l, p[0], p[1])   


def group_lines(lines, vertices):
    # Basic Approach
    #   Convert lines into (slope, intercept, length) form
    #   NOTE: actually using the reciprocal of the slope because most of our lines will be  close to vertical
    #   Bin into (slope, intercept) groups, 
    #   for each bin
    #       filter for items within envelope
    #       length-weighted average of slope, intercept
    #       generate line bound by ROI

    # Calculate bounds from ROI vertices

    y_max = vertices[0][0][1]
    y_min = vertices[0][1][1]

    x_min = vertices[0][0][0]
    x_max = vertices[0][3][0]
    x_mid = int((x_max + x_min) / 2)

    # Add all lines to dictionary using rounded slope and offset as the key. GROUP_SLOPE and GROUP_OFFSET
    # control the amount of rounding.

    clusters = {}

    for line in lines:
        (l, s, o) = len_slope_offset(line)
        key = (round(s * GROUP_SLOPE) / GROUP_SLOPE, round(o / GROUP_OFFSET) * GROUP_OFFSET)
        v = clusters.get(key, [])
        v.append((line, l, s, o))
        clusters[key] = v

    keys = list(clusters.keys())
    keys.sort()

    out = []

    # For each group of lines in the cluster, reject lines outside of envelope and then
    # perform a length-weighted average of slope + offset.

    for k in keys:
        v = clusters[k]
        v_lines = [l[0] for l in v]
        num = len(v)
        sum_len = sum([l[1] for l in v])
        if num >= GROUP_MIN_NUM and sum_len >= GROUP_MIN_SUM:
            s_sum = 0.0
            o_sum = 0.0
            l_sum = 0.0
            for item in v:
                (line, l, s, o) = item
                if GROUP_ITEM_ENABLE_ENVELOPE:
                    if o > x_mid:
                        if o < x_max * GROUP_LEFT_OFFSET_MIN or o > x_max * GROUP_LEFT_OFFSET_MAX:
                            #print('    left reject offset', s_avg, i_avg, sum_len)
                            continue
                        if s < GROUP_LEFT_SLOPE_MIN or s > GROUP_LEFT_SLOPE_MAX:
                            if DEBUG:
                                print('    left reject slope', s, o, l)
                            continue
                    else:
                        # print('  left')
                        if o < x_max * GROUP_RIGHT_OFFSET_MIN or o > x_max * GROUP_RIGHT_OFFSET_MAX:
                            #print('    right reject offset', s_avg, i_avg, sum_len)
                            continue
                        if s < GROUP_RIGHT_SLOPE_MIN or s > GROUP_RIGHT_SLOPE_MAX:
                            if DEBUG:
                                print('    right reject slope', s, o, l)
                            continue


                s_sum += (l * s)
                o_sum += (l * o)
                l_sum += l

            # No items left in this group due to filtering
            if l_sum == 0:
                continue

            s_avg = s_sum / l_sum
            o_avg = o_sum / l_sum

            # Calculate the line from y_min to y_max

            y0 = int(y_min)
            x0 = int((y0 * s_avg) + o_avg)
            y1 = int(y_max)
            x1 = int((y1 * s_avg) + o_avg)

            line = [[x0, y0, x1, y1]]

            # Add it to the output
            out.append(line)   
    return out

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return line_img

def process_image(img):
    # Keep the original image
    img_orig = img.copy()

    # Calculate the vertices based on the image size
    vertices = image_roi(img, ROI_TOP_WIDTH, ROI_TOP_HEIGHT)
    
    # Convert to grayscale
    img = helper.grayscale(img)

    # Run the Canny edge filter
    img = helper.canny(img, CANNY_LOW, CANNY_HIGH)

    if GAUSS_KERNEL > 0:
        # Blur the edge output
        img = helper.gaussian_blur(img, GAUSS_KERNEL)     
    
    # Mask the region of interest
    img = helper.region_of_interest(img, vertices)    
    
    # Generate candidate lines using the Hough transform
    lines = hough_lines(img, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LEN, HOUGH_MAX_LINE_GAP)    

    if GROUP_ENABLE:        
        # Group the candidate lines
        lines = group_lines(lines, vertices)

    # Draw final lines
    img_lines = draw_lines(img_orig, lines, LINE_COLOR, LINE_WIDTH)
    
    # Merge the original image with the generated lines.
    return helper.weighted_img(img_orig, img_lines, MERGE_ALPHA, MERGE_BETA, MERGE_LAMBDA)

def main(args):
    def press(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)

    fig = plt.figure(figsize=(8, 2 * len(args)))    
    fig.canvas.mpl_connect('key_release_event', press)   
    
    cols = 1
    rows = np.ceil(len(args) / float(cols))

    for i, arg in enumerate(args):
        title = os.path.splitext(os.path.basename(arg))[0]
        img = mpimg.imread(arg)
        a = fig.add_subplot(rows, cols, i + 1)
        a.set_title(title, fontsize=10)
        plt.axis('off')
        #print(arg)
        img_out = process_image(img)
        out_path = arg.replace('test_images/','test_images_output/')
        print(out_path)
        mpimg.imsave(out_path, img_out)
        plt.imshow(img_out)        

    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])