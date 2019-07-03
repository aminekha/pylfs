import cv2, os
import numpy as np
import matplotlib.image as mpimg
from keras.preprocessing.image import load_img, img_to_array


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return img_to_array(load_img(os.path.join(data_dir, image_file.strip()), grayscale=False)) /255.
    # return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    #image = rgb2yuv(image)
    return image


def choose_image(data_dir, center, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(data_dir, center, control, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, control = choose_image(data_dir, center, control)
    image, control = random_flip(image, control)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, control


def batch_generator(data_dir, image_paths, controls, batch_size, is_training):
    """
    Generate training image give image paths and associated control keys
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            image_path = image_paths[index]
            control = controls[index]
            
            image = load_image(data_dir, image_path)
            
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            # print("Images = ", images[i])
            # print()
            # print("control = \n", control)
            # print("Steers[i] = \n", steers[i])
            # steers[i] = control
            
            i += 1
            if i == batch_size:
                break
        # print("Images: ", images)
        # print("Control = ", control)
        yield images, control
        
# def generator(csv_path, batch_size, img_height, img_width, channels, augment=False):

#     ########################################################################
#     # The code for parsing the CSV (or loading the data files) should goes here
#     # We assume there should be two arrays after this:
#     #   img_path --> contains the path of images
#     #   annotations ---> contains the parsed annotaions
#     ########################################################################

#     n_samples = len(img_path)
#     batch_img = np.zeros((batch_size, img_width, img_height, channels))
#     idx = 0
#     while True:
#         batch_img_path = img_path[idx:idx+batch_size]
#         for i, p in zip(range(batch_size), batch_img_path):
#             img = image.load_img(p, target_size=(img_height, img_width))
#             img = image.img_to_array(img)
#             batch_img[i] = img

#         if augment:
#             ############################################################
#             # Here you can feed the batch_img to an instance of 
#             # ImageDataGenerator if you would like to augment the images.
#             # Note that you need to generate images using that instance as well
#             ############################################################

#         # Here we assume that the each column in annotations array
#         # corresponds to one of the outputs of our neural net
#         # i.e. annotations[:,0] to output1, annotations[:,1] to output2, etc. 
#         target = annotations[idx:idx+batch_size]
#         print("Target is = ", target)
#         batch_target = []
#         for i in range(annotations.shape[1]):
#             batch_target.append(target[:,i])

#         idx += batch_size
#         if idx > n_samples - batch_size:
#             idx = 0

#         yield batch_img, batch_target