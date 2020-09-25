from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QMessageBox
from PyQt5.uic import loadUi
from keras.models import load_model
from PIL import Image
import cv2
import os, sys
import numpy as np
from keras.preprocessing import image
import time 

model = load_model("SkinCancerModel.hdf5")

class SkinCancer(QDialog):
    def __init__(self):
        super(SkinCancer, self).__init__()
        loadUi("SkinCancer.ui" ,  self)
        self.takebtn.clicked.connect(self.take_function)
        self.receivebtn.clicked.connect(self.result_fun)
        self.receivebtn.setEnabled(False) 

    def take_function(self):
        #take image from cam and store
        cam = cv2.VideoCapture(0)
        Result = True
        while (Result):
            for i in range(5):
                ret, frame = cam.read()
                time.sleep(0.1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.__lastest_image = cv2.resize(image,(224,224))
            Result = False   
        cam.release()
        self.receivebtn.setEnabled(True) 


    def result_fun(self):
        #enter image to test 
        test_image = np.expand_dims(self.__lastest_image, axis=0)
        result = model.predict(test_image)
        if result[0][0] == 0:
            prediction = 'benign'
        else:
            prediction = 'malignant'
        self.receivebtn.setEnabled(False) 
        QMessageBox.about(self, "Result", f'the prediction result is {prediction}')


app = QApplication([])  
mainwindow = SkinCancer()
mainwindow.setFixedWidth(600)
mainwindow.setFixedHeight(800)
mainwindow.setWindowTitle('Skin Cancer App')
mainwindow.show()
app.exec_()


