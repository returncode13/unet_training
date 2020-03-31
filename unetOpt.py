import numpy as np
from comet_ml import Optimizer
from unet import *
from keras import Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys


def build_model():
    input_tensor=Input((256,256,1))
    model=unet(input_tensor,maxpool=False)
    return model




def create_data():    
    #create training data.
    X=np.random.randn(100,256,256,1)
    #generate reference data (label/true data)
    Y=0.8*X+2
    print(X.shape,Y.shape)
    #generate some validation data
    X_val=np.random.randn(100,256,256,1)
    Y_val=0.8*X_val+2
    print(X_val.shape,Y_val.shape)
    return (X,Y,X_val,Y_val)





def train(X_train=None,Y_train=None,validation_data=(None,None),batch=None,epochs=None):
    model=build_model()
    model.compile(optimizer=Adam(), loss="mse", metrics=["accuracy"])
    results=model.fit(X_train,Y_train,batch_size=batch,epochs=epochs,validation_data=validation_data)
    return results



opt=Optimizer(sys.argv[1],workspace="returncode13",project_name="UNET-training-01") 

(X,Y,X_val,Y_val)=create_data()

for experiment in opt.get_experiments():
    results=train(X,Y,batch=experiment.get_parameter('batch'),epochs=100,validation_data=(X_val,Y_val))
    experiment.log_metric('loss',results.history['loss'])
    experiment.log_metric('accuracy',results.history['accuracy'])

    
    