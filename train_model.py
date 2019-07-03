import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
#to save our model periodically as checkpoints for loading later
from keras.callbacks import ModelCheckpoint, TensorBoard
#popular optimization strategy that uses gradient descent 
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from models import nvidia_model
#helper class to define input shape and generate training images given image paths & steering angles
from utils import INPUT_SHAPE
from PIL import ImageGrab
from get_keys import key_check
import time
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

EPOCHS = 10
SAMPLES_PER_EPOCH = 20000
SAVE_BEST = True
LR = 1e-4
BATCH_SIZE = 64
TEST_SIZE = 0.2
DATA_DIR = "data/"
image_folder = os.path.join(DATA_DIR, "IMG")

z = [1,0,0,0] # forward
q = [0,1,0,0] # left
d = [0,0,1,0] # right
m = [0,0,0,1] # stop
zq = [1,1,0,0] # forward left
zd = [1,0,1,0] # forward right
nk = [0,0,0,0] # no keys

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [Z, Q, D, M, ZQ, ZD, NOKEY] boolean values.
    '''
    output = [0,0,0,0]

    if 'Z' in keys and 'Q' in keys:
        output = zq
    elif 'Z' in keys and 'D' in keys:
        output = zd
    elif 'Z' in keys:
        output = z
    elif 'M' in keys:
        output = m
    elif 'Q' in keys:
        output = q
    elif 'D' in keys:
        output = d
    else:
        output = nk
    return output


def get_data():
    # Create image folder
    try:
        os.mkdir(os.path.join("data", "IMG"))
    except:
        pass
    
    print("Collecting data after:")
    for i in list(range(10))[::-1]:
        print(i+1)
        time.sleep(1)
    
    paused = False
    print("Starting!")    
        
        
    
    # Fill csv file 
    # TODO: If this file doesn't exist create it and start new
    # If it exists keep it and append
    df = pd.DataFrame(columns=["image", "forward", "left", "right", "stop"])
    
    while True:
        if not paused:
            image_name = f"center_{datetime.today().strftime('%Y_%m_%d_%H_%M_%S_%f')}.jpg"
            
            screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            cv2.imshow("Window", cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(image_folder, image_name), screen)
            
            keys = key_check()
            output = keys_to_output(keys)
            # print(output)
            
            df = df.append({"image": os.path.join("IMG/", image_name), "forward": output[0], "left": output[1], "right": output[2], "stop": output[3]}, ignore_index=True)
            
            # print the number of images each 1000 steps
            folder_len = (len([name for name in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, name))]))
            if folder_len % 1000 == 0:
                print(folder_len, "Reached!")
            
            
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
            
            # print(df)
            
        check_paused = key_check()
        if "T" in check_paused:
            if paused:
                paused = False
                print("Unpaused!")
                time.sleep(1)
            else:
                print("Pausing")
                paused = True
                time.sleep(1)
                
    df.to_csv(os.path.join(DATA_DIR, "driving_log.csv"), index=False)
    


np.random.seed(0)
def load_data(data_dir, test_size):
    """
    Load training data and split it into training and validation set
    """
    # Get image and Multi-hot array from csv file
    data_df = pd.read_csv(os.path.join(os.getcwd(), data_dir, "driving_log.csv"), names=["image", "forward", "left", "right", "stop"])
    # Split data
    # TODO: Try to take more than one pic at time
    X = data_df["image"].values
    y = data_df[["forward", "left", "right", "stop"]].values
    
    # Split the data into training and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=test_size, random_state=0)
    train_df, test_df = train_test_split(data_df, test_size=0.2)
    
    print(len(X_train))
    print(len(y_train))
    print(len(X_valid))
    print(len(y_valid))
    
    print(X_train[0])
    print(y_train[0])
    print(X_valid[0])
    print(y_valid[0])
    return X_train, X_valid, y_train, y_valid, train_df, test_df

def train_model(model, save_best, lr, samples_per_epoch, epochs, data_dir, X_train, X_valid, y_train, y_valid, train_df, test_df):
    """
    Train the model
    """
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    #Saves the model after every epoch.
    #quantity to monitor, verbosity i.e logging mode (0 or 1), 
    #if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    #mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc, 
    #this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=save_best,
                                 mode='auto')

    #calculate the difference between expected steering angle and actual steering angle
    #square the difference
    #add up all those differences for as many data points as we have
    #divide by the number of them
    #that value is our mean squared error! this is what we want to minimize via
    #gradient descent
    model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=["accuracy"])
    
    # Create a TensorBoard instance with the path to the logs directory
    tensorboard = TensorBoard(log_dir='./logs', 
                              histogram_freq=0, 
                              batch_size=BATCH_SIZE, 
                              write_graph=False, 
                              write_grads=False, 
                              write_images=True, 
                              embeddings_freq=0, 
                              embeddings_layer_names=None, 
                              embeddings_metadata=None, 
                              embeddings_data=None, 
                              update_freq=10000)

    #Fits the model on data generated batch-by-batch by a Python generator.

    #The generator is run in parallel to the model, for efficiency. 
    #For instance, this allows you to do real-time data augmentation on images on CPU in 
    #parallel to training your model on GPU.
    #so we reshape our data into their appropriate batches and train our model simulatenously
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    y_col = ["forward", "left", "right", "stop"]
    
    train_generator = train_datagen.flow_from_dataframe(train_df,
                                                  directory=data_dir,
                                                  x_col="image",
                                                  y_col=y_col,
                                                  batch_size=BATCH_SIZE,
                                                  seed=42,
                                                  shuffle=False,
                                                  target_size=(66,200),
                                                  class_mode="other")
    valid_generator = test_datagen.flow_from_dataframe(test_df,
                                                directory=data_dir,
                                                x_col="image",
                                                y_col=y_col,
                                                batch_size=BATCH_SIZE,
                                                seed=42,
                                                shuffle=False,
                                                target_size=(66,200),
                                                class_mode="other")
    
    model.fit_generator(train_generator,
                        samples_per_epoch,
                        epochs,
                        max_q_size=1,
                        validation_data=valid_generator,
                        nb_val_samples=len(X_valid),
                        callbacks=[tensorboard],
                        verbose=1)

def main():
    # get_data()
    X_train, X_valid, y_train, y_valid, train_df, test_df = load_data(DATA_DIR, TEST_SIZE)
    model = nvidia_model(0.5)
    train_model(model, SAVE_BEST, LR, SAMPLES_PER_EPOCH, EPOCHS, DATA_DIR, X_train, X_valid, y_train, y_valid, train_df, test_df)
    
if __name__ == "__main__":
    main()
