import numpy as np
from PIL import ImageGrab
import cv2
import time
import os
from process_image import annotate_image
from grab_screen import grab_screen
from get_keys import key_check

def keys_to_output(keys):
    """
    Convert keys to a One-Hot encoder array
    [Z,Q,D,M]
    """
    output = [0, 0, 0, 0]
    if "Z" in keys:
        output[0] = 1
    if "Q" in keys:
        output[1] = 1
    if "D" in keys:
        output[2] = 1
    if "M" in keys:
        output[3] = 1
    return output

file_name = "training_data.npy"

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []

def main():
    for i in list(range(2))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    
    while True:   
        if not paused:    
            screen =  grab_screen(region=(0,40,800,640))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120))
            # resize for something acceptable for CNN
            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen, output])
            if len(training_data) % 100 == 0:
                print(f"{len(training_data)} Reached!")
            
            if len(training_data) % 1000 == 0:
                print(len(training_data))
                np.save(file_name, training_data)
                
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


        #print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        # #new_screen,original_image, m1, m2 = process_img(screen)
        
        # new_screen = annotate_image(screen)
        # cv2.imshow('Final view', new_screen)
        # #cv2.imshow('Module View',cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        
        #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
main()