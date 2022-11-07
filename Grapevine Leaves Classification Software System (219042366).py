"""
**************************************************************
Name: Abdulrahman Abu Raas
Student Number: 219042366
Project: Deep Learning Based Grapevine Leaves Classification
**************************************************************
"""

#Import Libraries:
import os
import cv2
import shutil
import pickle
import pathlib
import numpy as np
import splitfolders
import tkinter as tk
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tensorflow.keras import layers
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

#Constants:
IMG_SIZE = 256 #Image Size
B_SIZE = 32 #Batch Size

#Set colors of different GUI elements:
MainBColor = '#B6BE9D'
TitleColor = '#222518'
ClassColor = '#222518'
ClassifyButColor = '#222518'
UploadButColor = '#7A0000'

Scr_Res = '1280x720' #Resolution of app window

GrapevineIDModel = load_model('GVLImprovedCNN.h5') #Load trained and saved model to be used for classification

#Create "Upload" Button:
def Upload_Button():
    global UploadButton #global to be accessed by other functions
    #Create button with specific command (action to perform once clicked) and specify location, color, and size:
    UploadButton = Button(MainWin, text='Upload Leaf Image', command=lambda: Upload_Img(), padx=10, pady=5)
    UploadButton.configure(background=UploadButColor, foreground='#FFFFFF', font=('arial', 10, 'bold'))
    UploadButton.pack(side=BOTTOM, pady=30)

#Upload image function:
def Upload_Img():
    filePath = filedialog.askopenfilename() #allow user to choose file (image to be classified)
    imgUpload = Image.open(filePath) #Open the chosen image
    #Display image on application window:
    imgUpload.thumbnail(((MainWin.winfo_width()/2.2),(MainWin.winfo_height()/2.2)))
    img = ImageTk.PhotoImage(imgUpload)
    ImageContainer.configure(image=img)
    ImageContainer.image = img
    predLabel.configure(text='') #Set prediction label to empty (will be used later to store prediction of grapevine leaf from model)
    UserReqLabel.destroy() #Destroy the "Please upload an image" request label
    Classify_Button(filePath) #Create classify button
    UploadButton.destroy() #Destroy the upload button after image is uploaded
    trainButton.destroy() #Destroy the train button

#Create "Classify" Button:
def Classify_Button(filePath):
    global ClassifyButton #global to be accessed by other functions
    #Create button with specific command (action to perform once clicked) and specify location, color, and size:
    ClassifyButton = Button(MainWin, text='Classify', command=lambda: Classify_Image(filePath), padx=20, pady=5)
    ClassifyButton.configure(background=ClassifyButColor, foreground='white', font=('arial', 10, 'bold'))
    ClassifyButton.pack(side=BOTTOM, pady=30)

#Classify the chosen image:
def Classify_Image(filePath):
    ImgToClassify = load_img(filePath) #Load the image
    ImgToClassify = tf.keras.utils.img_to_array(ImgToClassify) #Convert to array
    ImgToClassify=cv2.resize(ImgToClassify, (IMG_SIZE,IMG_SIZE)) #Resize it to be able to go through model
    ImgToClassify = tf.expand_dims(ImgToClassify, 0) # Create a batch
    prediction = GrapevineIDModel.predict(ImgToClassify) #Predict the class of the image using the model
    score = tf.nn.softmax(prediction[0]) #get the score from prediction
    className = classes[np.argmax(score)] #Get the class of the image based on the prediction
    print(className) #print it to terminal
    predLabel.configure(foreground=ClassColor, text='Leaf belongs to the '+className+' class') #Write the class the grapevine leaf in the image belongs to in the prediction label
    ClassifyButton.destroy() #Destroy Button once clicked
    Train_Button() #Create the train button again
    Upload_Button() #Create the upload button again to allow user to choose another image.

#Create "Train" Button:
def Train_Button():
    global trainButton #global to be accessed by other functions
    #Create button with specific command (action to perform once clicked) and specify location, color, and size:
    trainButton = Button(text="Train", command=Train_Model_On_Data, padx=20, pady=5)
    trainButton.configure(background='black', foreground='#FFFFFF', font=('arial', 10, 'bold'))
    trainButton.pack(side=BOTTOM, pady=2)

#Create all necessary directories/folders within current directory to train the model:
def Create_Folders():
    #Check if directory (file) is available, and delete it if it is. Then make a new file with the path specified:
    if os.path.isdir(trainPath):
        shutil.rmtree(trainPath)
    os.mkdir(trainPath)

    #Check if directory (file) is available, and delete it if it is. Then make a new file with the path specified:
    if os.path.isdir(valPath):
        shutil.rmtree(valPath)
    os.mkdir(valPath)

    #Check if directory (file) is available, and delete it if it is. Then make a new file with the path specified:
    if os.path.isdir(testPath):
        shutil.rmtree(testPath)
    os.mkdir(testPath)

    #Create datasets with a split of 90/0/10 (train/valid/test)
    splitfolders.ratio(dataPath, output=currentDir, seed=12, ratio=(.9, 0, 0.1))
    shutil.rmtree(valPath) #Delete Validation folder (empty folder)

#Create/generate augmented images from training dataset:
def Gen_Aug_Images():
    #Check if directory (file) is available, and delete it if it is. Then make a new file with the path specified:
    if os.path.isdir(augPath):
        shutil.rmtree(augPath)
    os.mkdir(augPath)

    #Set the parameters for the data augmentation (different transformations to be applied, along with ranges):
    dataAug = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=50,
        width_shift_range=0.17,
        height_shift_range=0.17,
        channel_shift_range=50,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        brightness_range=[0.2, 1.7]
    )

    for imageFolder in os.listdir(trainPath): #Loop through each folder of training dataset
        os.mkdir(augPath+'/'+imageFolder+'/') #Create a new folder directory to store augmented images of class
        for file in os.listdir(trainPath+imageFolder+'/'): #Loop through all the images within each folder in training dataset
            img = load_img(trainPath+imageFolder+'/'+file) #Load an image from specific folder in directory
            x = img_to_array(img) #Convert image to an array (shape of array is: 3, width, height)
            x = x.reshape((1,) + x.shape) # Reshape array to 1,3,width, height (the 1 represents the batch size)
            
            count = 0 #Set count to keep track of how many times
            shutil.copy(trainPath+imageFolder+'/'+file, augPath+'/'+imageFolder) #Copy all images from original dataset to the augmented folder to have all images
            #Loop to create the augmented images based on dataaug:
            for batch in dataAug.flow(x, batch_size = 1, save_to_dir=augPath+'/'+imageFolder,save_prefix=imageFolder, save_format='png'): 
                count += 1
                if count > 15:
                    break #Break to stop the loop from generating further images

#Get the dataset path and set the paths required for training:
def Get_Set_Paths():
    #declare and initialize variables as global:
    global currentDir, dataPath, trainPath, valPath, testPath, augPath, checkpointPath
    dataPath = StringVar()
    fileName = filedialog.askdirectory() #Get dataset folder directory from user

    #Set Paths for each folder:
    trainPath = os.path.join(currentDir, r'train' + '\\')
    valPath = os.path.join(currentDir, r'val' + '\\')
    testPath = os.path.join(currentDir, r'test' + '\\')
    augPath = os.path.join(currentDir, r'Augmented_Images')
    checkpointPath = os.path.join(currentDir, r'Checkpoints')
    dataPath = fileName + '/'

    data_dir = pathlib.Path(dataPath) #Data directory
    ClassCount = len(list(data_dir.glob('*'))) #Get total number of classes within dataset
    ImageCount = len(list(data_dir.glob('*/*'))) #Get total number of images within dataset
    #Display the total number of classes and images within dataset:
    print('There is a total of {} classes in the grapevine leaves dataset.'.format(ClassCount))
    print('There is a total of {} images in the grapevine leaves dataset.'.format(ImageCount))

#Create the train/val datasets:
def Create_Train_Val_DS():
    global train_DS, val_DS, test_DS #global variables
    #Create train_DS using 80% of the augmented images:
    train_DS = tf.keras.preprocessing.image_dataset_from_directory(
        augPath,
        validation_split=0.2,
        subset='training',
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=B_SIZE
    )
    #Create val_DS using 20% of the augmented images:
    val_DS = tf.keras.preprocessing.image_dataset_from_directory(
        augPath,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=B_SIZE
    )
    #Create test_DS containing 100% of images in test dataset
    test_DS = tf.keras.preprocessing.image_dataset_from_directory(
    testPath,
    labels='inferred',
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=B_SIZE
)

#Build the Custom CNN Model:
def Build_Model():
    global class_names, num_classes, GVLModel
    class_names = train_DS.class_names #Get class names in dataset
    num_classes = len(class_names) #Get number of classes within dataset

    #Create and build the GVL Improved Custom CNN Model:
    GVLModel = Sequential([
        #First (input) layers where the images are normalized and the input shape is specified:
        layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'), #Convolutional layer, 16 filteres, and a kernal size of 3 with ReLu activation
        layers.MaxPooling2D(), #Maxpooling layer
        layers.Dropout(0.2), #Dropout with a probability of 0.2
        layers.Conv2D(32, 3, padding='same', activation='relu'), #Convolutional layer, 16 filteres, and a kernal size of 3 with ReLu activation
        layers.MaxPooling2D(), #Maxpooling layer
        layers.Conv2D(64, 3, padding='same', activation='relu'), #Convolutional layer, 16 filteres, and a kernal size of 3 with ReLu activation
        layers.MaxPooling2D(), #Maxpooling layer
        layers.Dropout(0.3), #Dropout with a probability of 0.3
        layers.Conv2D(128, 3, padding='same', activation='relu'), #Convolutional layer, 16 filteres, and a kernal size of 3 with ReLu activation
        layers.MaxPooling2D(), #Maxpooling layer
        layers.Conv2D(256, 3, padding='same', activation='relu'), #Convolutional layer, 16 filteres, and a kernal size of 3 with ReLu activation
        layers.MaxPooling2D(), #Maxpooling layer
        layers.Dropout(0.4), #Dropout with a probability of 0.4
        layers.Flatten(), #Flatten the 3 Dimension input to 1 Dimension output to be used by dense layer
        layers.Dense(256, activation='relu'), #Fully connected layer/dense layer with ReLu activation
        layers.Dropout(0.3), #Dropout with a probability of 0.3
        layers.Dense(128, activation='relu'), #Fully connected layer/dense layer with ReLu activation
        layers.Dropout(0.5), #Dropout with a probability of 0.5
        layers.Dense(num_classes) #Final prediction layer with number of classes as possible classification outputs
    ])

    #Compile the model and use the Adam optimizer with Sparse Categorical Crossentropy as the loss function:
    GVLModel.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    GVLModel.summary() #Print out a summary of the model (layers and output of each layer)

#Train the built model on the prepared dataset:
def Fit_Model():
    global history, epochs
    #Check if directory (file) is available, and delete it if it is. Then make a new file with the path specified:
    if os.path.isdir(checkpointPath):
        shutil.rmtree(checkpointPath)
    os.mkdir(checkpointPath)

    checkpointfile = checkpointPath+'wight-improve-{epoch:02d}-{val_accuracy:.2f}.h5' #Create path for saving checkpoints with best validation accuracy results
    #create a checkpoint:
    mdlcheckpoint = ModelCheckpoint(checkpointfile, monitor='val_accuracy', verbose=1,save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=0.5, patience=3, verbose=1, mode='max',min_lr=0.00001) #define when learning rate changes
    callbacklist = [mdlcheckpoint, reduce_lr] #create the callback list that after every epoch, saves if that epoch had better validation accuracy and adjust learning rate

    #Fit/train the model:
    epochs=100 #Set number of epochs to 100 (Number of times it goes through the training data)
    #Train/fit the model based on the training data and validate it using the validation data and store the history:
    history = GVLModel.fit(
    train_DS,
    validation_data=val_DS,
    epochs=epochs,
    callbacks=callbacklist
    )

#Save context:
def Save_All():
    #Save the list of class names:
    with open(currentDir+'\\'+'Classnameslist', 'wb') as file_pic:
        pickle.dump(class_names, file_pic)
    #Save new model that trained on new dataset:
    GVLModel.save(currentDir+'\\'+'GVLImprovedCNN.h5')
    #Save the model's history (useful in case user requires to analyze performance)
    with open(currentDir+'\\'+'ModelHistory', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

#Perform evaluation on the model:
def Evaluate_Model():
    #Plot the training and validation losses:
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(20, 10))
    plt.plot(epochs_range, train_loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc='upper right')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title('Training and Validation Loss (Improved GVL Custom CNN)')
    plt.grid()

    plt.show()

    #Plot the training and validation Accuracies:
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs_range = range(epochs)

    plt.figure(figsize=(20, 10))
    plt.plot(epochs_range, train_acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc='upper left')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title('Training and Validation Accuracy (Improved GVL Custom CNN)')
    plt.grid()

    plt.show()

    #Plot the confusion matrix and classification report using test dataset:
    y_test = []  #used to store the actual labels
    y_pred = []  #used to store the predicted labels

    #loop through the test dataset to get labels (actual and prediction):
    for img_b, lbl_b in test_DS:
        Prediction = GVLModel.predict(img_b, verbose=0) #Predict outputs (classes) using trained model
        y_test.append(lbl_b) #append the actual labels
        y_pred.append(np.argmax(Prediction, axis = - 1)) #append the predicted labels

    actualLabels = tf.concat([item for item in y_test], axis = 0)
    predLabels = tf.concat([item for item in y_pred], axis = 0)

    #print(actualLabels)
    #print(predLabels)

    confMat = confusion_matrix(actualLabels, predLabels) #Generate confusion matrix with actual and predicted values
    report = classification_report(actualLabels, predLabels, target_names=test_DS.class_names) #Generate classification report with actual and predicted values

    #Plot the confusion matrix:
    plt.figure(figsize=(9,9))
    sns.heatmap(confMat, annot=True, fmt='g', vmin=0, cbar=False, cmap='Reds')
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()

    #Print classification Report:
    print("\nClassification Report:\n", report)

    #Plot the confusion matrix and classification report using validation dataset:
    y_test = []  #used to store the actual labels
    y_pred = []  #used to store the predicted labels

    #loop through the test dataset to get labels (actual and prediction):
    for img_b, lbl_b in val_DS:
        Prediction = GVLModel.predict(img_b, verbose=0) #Predict outputs (classes) using trained model
        y_test.append(lbl_b) #append the actual labels
        y_pred.append(np.argmax(Prediction, axis = - 1)) #append the predicted labels

    actualLabels = tf.concat([item for item in y_test], axis = 0)
    predLabels = tf.concat([item for item in y_pred], axis = 0)

    confMat = confusion_matrix(actualLabels, predLabels) #Generate confusion matrix with actual and predicted values
    report = classification_report(actualLabels, predLabels, target_names=val_DS.class_names) #Generate classification report with actual and predicted values

    #Plot the confusion matrix:
    plt.figure(figsize=(9,9))
    sns.heatmap(confMat, annot=True, fmt='g', vmin=0, cbar=False, cmap='Reds')
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()

    #Print classification Report:
    print("\nClassification Report:\n", report)

#Training model on given dataset:
def Train_Model_On_Data():
    Get_Set_Paths() #Set the paths for all required directories
    Create_Folders() #Split the dataset and create required train and test folders
    Gen_Aug_Images() #Generate the augmented images from dataset
    Create_Train_Val_DS() #Split the augmneted images to train and validation datasets and make the original test dataset
    Build_Model() #Create the proposed model (GVL Improved Custom CNN)
    Fit_Model() #Train the model built on the training dataset
    Save_All() #Save context so application can use new knowledge to classify images from new dataset even after software shutdown
    Evaluate_Model() #Model Evaluation to view performance

#Setup all the GUI elements:
def Application_GUI():
    global MainWin, ImageContainer, predLabel, UserReqLabel, classes, currentDir
    
    currentDir = os.getcwd() #Get current directory

    MainWin = tk.Tk() #Create object
    MainWin.geometry(Scr_Res) #Setup applcation's window size
    MainWin.title("Grapevine Leaves Image Classifier V2.4") #Set the title of the application window
    MainWin.configure(background=MainBColor) #Choose background color of the main applicaiton window

    #Create the main title and display it:
    titleLabel = Label(MainWin, text="Grapevine Leaves Image Classifier", font=('arial', 45, 'bold', 'underline'), pady=30) #Choose font (changed the order of args)
    titleLabel.configure(background=MainBColor, foreground=TitleColor) #Choose background and foreground colors
    titleLabel.pack(side=TOP)

    #Create the information label and display it:
    infoLabel = Label(MainWin, text="Mabe by: Abdulrahman Abu Raas | Version: 2.4", font=('arial', 7)) #Choose font (changed the order of args)
    infoLabel.configure(background=MainBColor, foreground=TitleColor) #Choose background and foreground colors
    infoLabel.pack(side=BOTTOM, pady=2)

    #Load the class names list (this contains the most recent class names being classified by software system):
    with open(currentDir+'\\'+'Classnameslist', "rb") as file_pic:
        classes = pickle.load(file_pic)

    Train_Button() #create the train button
    Upload_Button() #create the upload button

    #create the image container that the uploaded image will be displayed in:
    ImageContainer = Label(MainWin, background=MainBColor) 
    ImageContainer.pack(side = TOP, expand=True)

    #Create the prediction label that will contain the predicted class/result from classification:
    predLabel = Label(MainWin, background=MainBColor, font = ('arial', 30, 'bold'))
    predLabel.pack(side=TOP, expand=True)

    #Create user request label and display it:
    UserReqLabel = Label(MainWin, text="Please upload an image", font=('arial', 20, 'bold'), pady=40) #Choose font (changed the order of args)
    UserReqLabel.configure(background=MainBColor, foreground=TitleColor) #Choose background and foreground colors
    UserReqLabel.pack(side=BOTTOM)

    MainWin.mainloop() #Create application window

#Start the application:
Application_GUI()