
# use `pythonw p1.py` on OSX

import sys, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import helper

CANNY_LOW = 100
CANNY_HIGH = 250

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
HOUGH_THRESHOLD = 50
# minimum number of pixels making up a line
HOUGH_MIN_LINE_LEN = 20
# maximum gap in pixels between connectable line segments
HOUGH_MAX_LINE_GAP = 5

LINE_COLOR = [0, 255, 0]
LINE_WIDTH = 4

GROUP_OFFSET = 500.0
GROUP_SLOPE = 0.5
GROUP_MIN_NUM = 1
GROUP_MIN_SUM = 1

GROUP_LEFT_OFFSET_MIN = 0.0
GROUP_LEFT_OFFSET_MAX = 0.1
GROUP_LEFT_SLOPE_MIN = 1.0
GROUP_LEFT_SLOPE_MAX = 2.0

GROUP_RIGHT_OFFSET_MIN = 0.9
GROUP_RIGHT_OFFSET_MAX = 1.1
GROUP_RIGHT_SLOPE_MIN = -2.0
GROUP_RIGHT_SLOPE_MAX = -1.0



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
        l = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        p = np.polyfit([y1, y2], [x1, x2], 1)
        # Only process first segment of line
        return (l, p[0], p[1])   


def filter_lines(lines, vertices):
    # Basic Approach
    #   Convert lines into (slope, intercept, length) form
    #   Bin into (slope, intercept)
    #   Search grid for boxes with sum(length) above threshold
    #     - Weighted average (slope, intercept) => line    
    #     - Linear regression

    # TODO - group based on positive / negative slope, linear regression of each
    # TODO - draw line based on slope + intercept
    # TODO - return line list with line type (LINE_SOLID, LINE_DASHED, LINE_LEFT, LINE_RIGHT, LINE_WHITE, LINE_YELLOW)

    y_max = vertices[0][0][1]
    y_min = vertices[0][1][1]

    x_min = vertices[0][0][0]
    x_max = vertices[0][3][0]
    x_mid = int((x_max + x_min) / 2)

    clusters = {}

    for line in lines:
        (l, s, i) = len_slope_intercept(line)
        key = (round(s * GROUP_SLOPE) / GROUP_SLOPE, round(i / GROUP_OFFSET) * GROUP_OFFSET)
        #print(line, l, s, i, key)
        v = clusters.get(key, [])
        v.append((line, l, s, i))
        clusters[key] = v

    keys = list(clusters.keys())
    keys.sort()

    #print("groups:", len(keys))

    out = []

    for k in keys:
        v = clusters[k]
        v_lines = [l[0] for l in v]
        num = len(v)
        sum_len = sum([l[1] for l in v])
        #print(k, len(v), sum_len)
        if num >= GROUP_MIN_NUM and sum_len >= GROUP_MIN_SUM:
            #print(k)
            s_sum = 0.0
            i_sum = 0.0
            l_sum = 0.0
            # y_min = 0.0
            # y_max = 0.0
            for item in v:
                (line, l, s, i) = item
                #print("  %s %4.2f %f %f" % (line, l, s, i))
                s_sum += (l * s)
                i_sum += (l * i)
                l_sum += l
                (x1, y1, x2, y2) = line[0]

                dx = x2 - x1
                dy = y2 - y1
                m = dx / dy

                #print("      dx = %f dy = %f m = %f" % (dx, dy, m))

                # if y1 < y_min or y_min == 0.0:
                #     y_min = y1
                # if y2 < y_min:
                #     y_min = y2
                # if y1 > y_max:
                #     y_max = y1
                # if y2 > y_max:
                #     y_max = y2

            s_avg = s_sum / l_sum
            i_avg = i_sum / l_sum
            #print("  %f %f (%f %f)" % (s_avg, i_avg, y_min, y_max))
            
            # filter for sane s_avg

            if i_avg > x_mid:
                if i_avg < x_max * GROUP_RIGHT_OFFSET_MIN or i_avg > x_max * GROUP_RIGHT_OFFSET_MAX:
                    print('    right reject offset', s_avg, i_avg)
                    continue
                if s_avg < GROUP_RIGHT_SLOPE_MIN or s_avg > GROUP_RIGHT_SLOPE_MAX:
                    print('    right reject slope', s_avg, i_avg)
                    continue
            else:
                if i_avg < x_max * GROUP_LEFT_OFFSET_MIN or i_avg > x_max * GROUP_LEFT_OFFSET_MAX:
                    print('    left reject offset', s_avg, i_avg)
                    continue
                if s_avg < GROUP_LEFT_SLOPE_MIN or s_avg > GROUP_LEFT_SLOPE_MAX:
                    print('    left reject slope', s_avg, i_avg)
                    continue


            # y = mx + b

            line = v[0][0][0]            
            
            # y0 = int(line[1])
            # x0 = int((y0 * s_avg) + i_avg)
            # y1 = int(line[3])
            # x1 = int((y1 * s_avg) + i_avg)

            y0 = int(y_min)
            x0 = int((y0 * s_avg) + i_avg)
            y1 = int(y_max)
            x1 = int((y1 * s_avg) + i_avg)

            line = [[x0, y0, x1, y1]]
            #print(" line:", line)
            out.append(line)   
            #out.extend(v_lines)
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
    #print(vertices)
    img = helper.grayscale(img)
    img = helper.canny(img, CANNY_LOW, CANNY_HIGH)
    img = helper.gaussian_blur(img, GAUSS_KERNEL)     
    
    img = helper.region_of_interest(img, vertices)    
    #return img
    lines = hough_lines(img, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LEN, HOUGH_MAX_LINE_GAP)    
    lines = filter_lines(lines, vertices)

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
        #print(arg)
        img_out = process_image(img)
        plt.imshow(img_out)        

    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])