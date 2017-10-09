import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import helper

def display_image():
    image = mpimg.imread('test_images/solidWhiteRight.jpg')
    #printing out some stats and plotting
    print('This image is:', type(image), 'with dimensions:', image.shape)
    plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
    plt.show()

def display_image_gray():
    image = mpimg.imread('test_images/solidWhiteRight.jpg')
    image = helper.grayscale(image)
    plt.imshow(image, cmap='gray')  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
    plt.show()
    
def display_canny():
    image = mpimg.imread('test_images/solidWhiteRight.jpg')
    image = helper.canny(image, 150, 200)
    plt.imshow(image, cmap='gray')
    plt.show()

def display_roi():
    image = mpimg.imread('test_images/solidWhiteRight.jpg')

    imshape = image.shape
    vertices = np.array([[
        (0,imshape[0]),
        (imshape[1]*0.45, imshape[0] * 0.55), 
        (imshape[1]*0.55, imshape[0] * 0.55), 
        (imshape[1],imshape[0])
    ]], dtype=np.int32)

    image = helper.region_of_interest(image, vertices)
    plt.imshow(image, cmap='gray')
    plt.show()

def display_lines():
    image = mpimg.imread('test_images/solidWhiteRight.jpg')
    lines = [
        [
            [0, 0, 100, 100],
            [10, 0, 110, 100],
        ],
        [
            [20, 0, 120, 100]
        ],
    ]
    # for line in lines:
    #     print("line", line)
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    helper.draw_lines(image, lines)
    plt.imshow(image)
    plt.show()

def display_hough_lines():
    image = mpimg.imread('test_images/solidWhiteRight.jpg')
    image = helper.canny(image, 150, 200)

    rho = 1 # distance resolution in pixels of the Hough grid
    theta = 2 * (np.pi/180) # angular resolution in radians of the Hough grid
    threshold = 1     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
    # line_image = np.copy(image)*0 # creating a blank to draw lines on
    
    image = helper.hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap)
    plt.imshow(image)
    plt.show()

def display_weighted_hough_lines():
    image = mpimg.imread('test_images/solidWhiteRight.jpg')
    canny = helper.canny(image.copy(), 150, 200)

    rho = 1 # distance resolution in pixels of the Hough grid
    theta = 2 * (np.pi/180) # angular resolution in radians of the Hough grid
    threshold = 1     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
    # line_image = np.copy(image)*0 # creating a blank to draw lines on
    
    hough = helper.hough_lines(canny, rho, theta, threshold, min_line_len, max_line_gap)

    combined = helper.weighted_img(hough, image)

    plt.imshow(combined)
    plt.show()
    

def main():
    display_weighted_hough_lines()

if __name__ == '__main__':
    main()