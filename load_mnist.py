import numpy as np
import imageio
import glob

def load_mnist():
    # Loads the MNIST dataset from png images
 
    NUM_LABELS = 10        
    # create list of image objects
    test_images = []
    test_labels = []    
    
    for label in range(NUM_LABELS):
        for image_path in glob.glob("/Users/ZwYu/Desktop/DL/A1/MNIST/Test/" + str(label) + "/*.png"):
            image = imageio.imread(image_path)
            test_images.append(image)
            letter = [0 for _ in range(0,NUM_LABELS)]    
            letter[label] = 1
            test_labels.append(letter)  
            
    # create list of image objects
    train_images = []
    train_labels = []    
    
    for label in range(NUM_LABELS):
        for image_path in glob.glob("/Users/ZwYu/Desktop/DL/A1/MNIST/Train/" + str(label) + "/*.png"):
            image = imageio.imread(image_path)
            train_images.append(image)
            letter = [0 for _ in range(0,NUM_LABELS)]    
            letter[label] = 1
            train_labels.append(letter)                  
            
    # X_train= np.array(train_images).reshape(-1,784)/255.0
    X_train= np.array(train_images).reshape(-1,784)
    Y_train= np.array(train_labels)
    # X_test= np.array(test_images).reshape(-1,784)/255.0
    X_test= np.array(test_images).reshape(-1,784)
    Y_test= np.array(test_labels)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255.
    X_test /= 255.
    
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = load_mnist()