import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Input, Flatten
import matplotlib.pyplot as plt
import pygame
from PIL import Image
import cv2
from resizeimage import resizeimage
import time
import pandas as pd
import tkinter as tk


np.random.seed(42)

#Uncomment following lines if you want to train the network again!
#Download the dataset from https://www.kaggle.com/tkk233/recognize-hand-written-alphabets/data?scriptVersionId=3758333

'''
data = pd.read_csv('A_Z Handwritten Data.csv')

X = data.iloc[:, 1:].values
y = data.iloc[:,0].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.027, random_state = 0)
X_train = np.reshape(X_train, [X_train.shape[0],28,28,1])
X_test = np.reshape(X_test, [X_test.shape[0],28,28,1])
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape)
'''
num_classes= 26
input_shape = (28,28,1)

def alphaModel(input_shape = (28, 28, 1)):
    X_input = Input(input_shape)
    X = Conv2D(filters = 32, kernel_size = (5, 5), strides = (1,1), padding = 'same',activation='relu')(X_input)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(X)

    X = Conv2D(filters = 64, kernel_size = (5,5), strides = (1,1), padding = 'same', activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(X)
    X = Flatten()(X)
    X = Dense(1024, activation='relu', kernel_initializer='TruncatedNormal',bias_initializer='zeros')(X)
    X = Dense(num_classes, activation='softmax', kernel_initializer='TruncatedNormal',bias_initializer='zeros')(X)
    
    model = Model(inputs = X_input, outputs = X, name='alphaModel')

    return model

model = alphaModel(input_shape = (28,28,1))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])




#=============Uncomment the following lines if you want to train the network again!
'''X_train = X_train/255
X_test = X_test/255
hist = model.fit(X_train, y_train,
          batch_size=512,
          epochs=10,
          validation_data=(X_test, y_test), 
          )
## Don't forget to set the file explorer as current directory in spyder!
model.save_weights("weights.h5")
'''
#==================================================================================
model.load_weights("weights.h5")


score = 0.995923
score2 = 0.99870858
#Uncomment the following lines if you want to evaluate manually
#score = model.evaluate(X_test, y_test)[1]
#score2 = model.evaluate(X_train, y_train)[1]
print("Train Accuracy: "+str(score2*100)+"%")
print("Test Accuracy: "+str(score*100)+"%")

#===========================End of recognition part============================

pygame.init()

display_width = 300
display_height = 300
radius = 10

black = (0,0,0) # RGB
white = (255,255,255)

#=================tKinter part=================================================
textWidth, textHeight = 400,200
root = tk.Tk()
root.title('Text Box')
root.minsize(width=textWidth,height=textHeight)
root.maxsize(width=textWidth,height=textHeight)
text = tk.Text(root,width=textWidth,height=textHeight)
text.pack()


root.update()

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Drawing pad')

def alphaPredict(gameDisplay):
    #Processing the image
    data = pygame.image.tostring(gameDisplay, 'RGBA')
    img = Image.frombytes('RGBA', (display_width, display_height), data)
    img = resizeimage.resize_cover(img, [28, 28])
    imgobj = np.asarray(img)
    imgobj = cv2.cvtColor(imgobj, cv2.COLOR_RGB2GRAY)
    (_, imgobj) = cv2.threshold(imgobj, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #Predicting
    imgobj = imgobj/255
    b = model.predict(np.reshape(imgobj,[1, imgobj.shape[0], imgobj.shape[1],1]))
    ans = np.argmax(b)
    return ans

def textObjects(text, font):
	textSurface = font.render(text, True, white)
	return textSurface, textSurface.get_rect()

def message_display(text, locx, locy,size):
	largeText = pygame.font.Font('freesansbold.ttf', size) # Font(font name, font size)
	TextSurf, TextRec = textObjects(text, largeText)
	TextRec.center = (locx,locy)
	gameDisplay.blit(TextSurf, TextRec)
	pygame.display.update()

def resetScreen(gameDisplay):
    gameDisplay.fill(black)
    pygame.display.flip()

def saveText():
    t = text.get(0.0, tk.END)
    f = open('output.txt','w')
    f.write(t)
    f.close()
    
def gameLoop():
    gameExit = False
    resetScreen(gameDisplay)
    tick = tock = 0
    startDraw = False
    RMBDown = False
    MMBDown = False
    while not gameExit:
        if tock - tick >= 1 and startDraw:
            predVal = alphaPredict(gameDisplay)
            gameDisplay.fill(black)
            message_display("Predicted Alphabet: "+str(chr(ord('A')+predVal)), int(display_width/2), int(display_height/2), 20)
            text.insert(tk.INSERT,str(chr(ord('A')+predVal)))
            time.sleep(1)#sleep for 1 seconds
            resetScreen(gameDisplay)
            tick = 0
            tock = 0
            startDraw = False
            continue
        
        tock = time.clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True
        if pygame.mouse.get_pressed()[0]:
            spot = pygame.mouse.get_pos()
            pygame.draw.circle(gameDisplay,white,spot,radius)
            pygame.display.flip()
            tick = time.clock()
            startDraw = True
        if pygame.mouse.get_pressed()[2] and not RMBDown:#deletion
            gameDisplay.fill(black)
            pygame.display.flip()
            tick = tock = 0
            startDraw = False
            text.delete("insert -1 chars", "insert")
            RMBDown = True
        elif RMBDown and not pygame.mouse.get_pressed()[2]:
            RMBDown = False
        if pygame.mouse.get_pressed()[1] and not MMBDown:#space
            text.insert(tk.INSERT,' ')
            root.update()
            MMBDown = True
            message_display("Space Added", int(display_width/2), int(display_height/2), 20)
            time.sleep(1)
            resetScreen(gameDisplay)
        elif MMBDown and not pygame.mouse.get_pressed()[1]:
            MMBDown = False
        root.update()
            
gameLoop()
saveText()
root.destroy()   
pygame.quit()