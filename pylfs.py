import numpy as np
from PIL import ImageGrab
import cv2
import time
from controls import PressKey, Z, Q, D, S

def draw_lines(img, lines):
    """
    This function will draw the edges of the road
    """
    try:
        for lines in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)
    except:
        pass 

def roi(img, vertices):
    """
    Region Of Interest:
    This function will remove all the unecessary lines. 
    So the module will only see the road lines.
    """
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_img(original_image):
    # convert to gray
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    # Blur the view
    processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)
    vertices = np.array([[100, 500], [10,300], [300,200], [500,200], [800,300], [800,500]], np.int32)
    # remove unecessary elements from the view
    processed_img = roi(processed_img, [vertices])
    # draw road edges using Hough algorithm
    # TODO: trick parameters to improve edge detection
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, 20, 15)
    draw_lines(processed_img, lines)
    
    return processed_img

last_time = time.time()
while(True):
    screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
    new_screen = process_img(screen)
    last_time = time.time()
    cv2.imshow('window', new_screen)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break