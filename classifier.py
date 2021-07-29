import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X,y=fetch_openml('mnist_784',version=1,return_X_y=True)

xTrain,xTest,yTrain,yTest=train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)
xTrainScaled=xTrain/255.0
xTestScaled=xTest/255.0
clf=LogisticRegression(solver="saga",multi_class="multinomial").fit(xTrainScaled,yTrain)

def getPrediction(image):
    imagePillow=Image.open(image)
    imageBw=imagePillow.convert("L")
    imageResize=imageBw.resize((28,28),Image.ANTIALIAS)
    pixelFilter=20
    minPixel=np.percentile(imageResize,pixelFilter)
    imageScaled=np.clip(imageResize-minPixel,0,255)
    maxPixel=np.max(imageResize)
    imageScaled=np.asarray(imageScaled)/maxPixel
    testSample=np.array(imageScaled).reshape(1,784)
    prediction=clf.predict(testSample)
    return prediction[0]
    