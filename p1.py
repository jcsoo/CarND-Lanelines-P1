import sys, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import helper

CANNY_LOW = 100
CANNY_HIGH = 200

GAUSS_KERNEL = 3

# fractional width of parallelogram top
ROI_TOP_WIDTH = 0.30
# fractional height of parallelogram
ROI_TOP_HEIGHT = 0.35

# distance resolution in pixels of the Hough grid
HOUGH_RHO = 1
# angular resolution in radians of the Hough grid
HOUGH_THETA = 2 * (np.pi / 180)
# minimum number of votes (intersections in Hough grid cell)
HOUGH_THRESHOLD = 4
# minimum number of pixels making up a line
HOUGH_MIN_LINE_LEN = 20
# maximum gap in pixels between connectable line segments
HOUGH_MAX_LINE_GAP = 10

LINE_COLOR = [255, 0, 0]
LINE_WIDTH = 1

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

def len_slope_intercept(line):
    si = []
    for x1, y1, x2, y2 in line:
        #si.append(abs(x2-x1) / abs(y2-y1))
        l = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        p = np.polyfit([y1, y2], [x1, x2], 1)
        # Only process first segment of line
        return (l, p[0], p[1])   


def filter_lines(lines):
    # Basic Approach
    #   Convert lines into (slope, intercept, length) form
    #   Bin into (slope, intercept)
    #   Search grid for boxes with sum(length) above threshold
    #     - Weighted average (slope, intercept) => line    
    #     - Linear regression

    # TODO - group based on positive / negative slope, linear regression of each
    # TODO - draw line based on slope + intercept
    # TODO - return line list with line type (LINE_SOLID, LINE_DASHED, LINE_LEFT, LINE_RIGHT, LINE_WHITE, LINE_YELLOW)


    clusters = {}

    for line in lines:
        (l, s, i) = len_slope_intercept(line)
        key = (round(s * 10.0), round(i / 40.0))
        #print(line, l, s, i, key)
        v = clusters.get(key, [])
        v.append((line, l, s, i))
        clusters[key] = v

    keys = list(clusters.keys())
    keys.sort()

    out = []

    for k in keys:
        v = clusters[k]
        v_lines = [l[0] for l in v]
        num = len(v)
        sum_len = sum([l[1] for l in v])
        #print(k, len(v), sum_len)
        if num > 1 and sum_len > 100:
            #print(k, v)
            out.extend(v_lines)
    return out

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return line_img

def process_image(img):    
    img_orig = img.copy()
    vertices = image_roi(img, ROI_TOP_WIDTH, ROI_TOP_HEIGHT)

    img = helper.grayscale(img)
    img = helper.canny(img, CANNY_LOW, CANNY_HIGH)
    img = helper.gaussian_blur(img, GAUSS_KERNEL) 
    
    img = helper.region_of_interest(img, vertices)    
    
    lines = hough_lines(img, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LEN, HOUGH_MAX_LINE_GAP)
    lines = filter_lines(lines)

    img_lines = draw_lines(img_orig, lines, LINE_COLOR, LINE_WIDTH)
    
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
        # plt.axis('off')
        img_out = process_image(img)
        plt.imshow(img_out)        

    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])