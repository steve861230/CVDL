from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2 
import numpy as np
from Ui_GUI import Ui_Form
from PIL import Image
import matplotlib.pyplot as plt
from lenet import my_model
from tensorflow.examples.tutorials.mnist import input_data

dog = './images/dog.bmp'
color = './images/color.png'
QR = './images/QR.png'
school = './images/School.jpg'
OT = './images/OriginalTransform.png'
PT = './images/OriginalPerspective.png'


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.loadButton.clicked.connect(self.load_image)
        self.ui.colorButton.clicked.connect(self.color_conversion)
        self.ui.flippingButton.clicked.connect(self.flipping)
        self.ui.blendingButton.clicked.connect(self.blending)
        self.ui.gtButton.clicked.connect(self.global_threshold)
        self.ui.ltButton_3.clicked.connect(self.local_threshold)
        self.ui.RSTButton.clicked.connect(self.RST)
        self.ui.ptButton.clicked.connect(self.PT)
        self.ui.showButton.clicked.connect(self.show_data)
        self.ui.paraButton.clicked.connect(self.show_hyper)
        self.model = my_model(32,10,1,0.001,28,28)
        self.ui.epochButton.clicked.connect(self.train_1_epoch)
        self.ui.inferenceButton.clicked.connect(self.predict)
        self.ui.gaussianButton.clicked.connect(self.gaussian)
        self.ui.sebelxButton.clicked.connect(self.sobel_x)
        self.ui.sebelyButton.clicked.connect(self.sobel_y)
        self.ui.magButton.clicked.connect(self.magnitude)

    def load_image(self):
        self.img_dog = cv2.imread(dog)
        cv2.imshow('dog', self.img_dog)
        
    def color_conversion(self):
        self.img_color = cv2.imread(color)
        self.red = self.img_color[:,:,2]
        self.green = self.img_color[:,:,1]
        self.blue = self.img_color[:,:,0]
        self.img_color[:,:,1] = self.red
        self.img_color[:,:,2] = self.blue
        self.img_color[:,:,1] = self.green
        cv2.imshow('rgb',self.img_color)

    def flipping(self):
        self.img_dog = cv2.imread(dog)
        self.img_dog = cv2.flip(self.img_dog, 1)
        cv2.imshow('dog', self.img_dog)
    def nothing(self,x):
        pass
    def blending(self,x):
        cv2.namedWindow('blend')
        self.img_dog = cv2.imread(dog)
        self.img_dog_r = cv2.flip(self.img_dog, 1)
        cv2.createTrackbar('bar','blend',0,100,self.nothing)
        while(1):
            self.b = cv2.getTrackbarPos('bar','blend')
            self.blend_dog = cv2.addWeighted(self.img_dog,self.b/100 , self.img_dog_r, (100-self.b)/100, 0)
            cv2.imshow('blend', self.blend_dog)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
    def global_threshold(self):
        self.img_QR = cv2.imread(QR, 0)
        self.img_QR = cv2.adaptiveThreshold(self.img_QR,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,75,10)
        cv2.imshow('global_threshold', self.img_QR)

    def local_threshold(self):
        self.img_QR = cv2.imread(QR,0)
        self.img_QR = cv2.adaptiveThreshold(self.img_QR,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,75,10)
        cv2.imshow('local_threshold', self.img_QR)
    def RST(self): 
        self.img_OT = cv2.imread(OT,1)
        self.row , self.col = self.img_OT.shape[:2]
        self.angle = float(self.ui.lineEdit.text())
        self.scale = float(self.ui.lineEdit_2.text())
        self.Tx = float(self.ui.lineEdit_3.text())
        self.Ty = float(self.ui.lineEdit_4.text())
        self.rst = cv2.getRotationMatrix2D((130+self.Tx,125+self.Ty),self.angle,self.scale)
        self.img_OT = cv2.warpAffine(self.img_OT,self.rst,(self.col,self.row))
        cv2.imshow('RST', self.img_OT)

    def PT(self):
        self.i = 0
        self.img_PT = cv2.imread(PT,1)
        self.row , self.col = self.img_PT.shape[:2]
        self.pts2 =  np.float32([[20,20],[450,20],[20,450],[450,450]])
        self.pts = [(0,0),(0,0),(0,0),(0,0)]
        cv2.namedWindow('PT')
        cv2.setMouseCallback('PT',self.draw_circle)
        while(1):
            cv2.imshow('PT', self.img_PT)
            if self.i == 4:
                break
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                break
        self.pts1 = np.float32([\
			[self.pts[0][0],self.pts[0][1]],\
			[self.pts[1][0],self.pts[1][1]],\
			[self.pts[2][0],self.pts[2][1]],\
			[self.pts[3][0],self.pts[3][1]] ])    
        self.M = cv2.getPerspectiveTransform(self.pts1,self.pts2)
        self.dst = cv2.warpPerspective(self.img_PT,self.M,(450,450))
        cv2.imshow('PT', self.dst)

    def draw_circle(self,event,x,y,flags,param):    
        if event == cv2.EVENT_LBUTTONDBLCLK and self.i <= 4:
            cv2.circle(self.img_PT,(x,y),10,(255,0,0),-1)
            self.pts[self.i] = (x,y)
            self.i += 1

    def show_hyper(self):
        self.model.print_parameters()

    def train_1_epoch(self):
        self.model.load()
        self.model.build()
        self.model.train()
        self.model.plot()

    def show_data(self):
        mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
        mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)


        trainimg = mnist.train.images
        trainlabel = mnist.train.labels
        nsample = 1
        randidx = np.random.randint(trainimg.shape[0], size=nsample)

        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            curr_img   = np.reshape(trainimg[i, :], (28, 28)) # 28 by 28 matrix 
            curr_label = np.argmax(trainlabel[i, :] ) # Label
            plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
            plt.title("" + str(i + 1) + "th Training Data " 
                                    + "Label is " + str(curr_label))
        plt.show()    
    
    def predict(self):
        self.result = np.zeros(10)
        self.result[4] = 20
        self.num = ('0','1','2','3','4','5','6','7','8','9')
        self.y_pos = np.arange(len(self.num))
        self.model.load()
        self.model.build()
        self.model.load_weight()
        self.index = int(self.ui.lineEdit_index.text())
        self.image = self.model.x_test[self.index,:,:,0]
        print(self.model.x_test.shape)
        plt.imshow(self.image, cmap='binary')
        self.Y_pred = self.model.predict(self.image)
        print(np.argmax(self.Y_pred))
        plt.bar(self.y_pos, self.result, align='center', alpha=0.5)
        plt.xticks(self.y_pos, self.num)

        plt.show()
    
    def gaussian (self):
       
        self.gaussian_kernel = np.array([[0.045,0.122,0.045],
                                        [0.122,0.332,0.122],
                                        [0.045,0.122,0.045]])

        #Normalization
        self.gaussian_kernel = self.gaussian_kernel / self.gaussian_kernel.sum()
        self.img_school = cv2.imread(school)
        self.img_school = cv2.cvtColor(self.img_school,cv2.COLOR_BGR2GRAY) #灰階
        self.new_school = self.new_school = self.convolution2d(self.img_school, self.gaussian_kernel, 0)
        # cv2.imshow('gaussian', self.new_school)
        plt.imshow(self.new_school, cmap=plt.get_cmap('gray'))
        plt.show()

    def convolution2d(self, image, kernel, bias):
        self.m, self.n = kernel.shape
        if (self.m == self.n):
            self.y, self.x = image.shape
            self.y = self.y - self.m + 1
            self.x = self.x - self.m + 1
            new_image = np.zeros((self.y,self.x))
            for i in range(self.y):
                for j in range(self.x):
                    new_image[i][j] = np.sum(image[i:i+self.m, j:j+self.m]*kernel) + bias
        return new_image

    def sobel_x(self):
        self.sobelx_kernel = np.array([[-1,0,1],
                                        [-2,0,2],
                                        [-1,0,1]])
        self.sx_school = self.convolution2d(self.new_school, self.sobelx_kernel, 0)
        plt.imshow(self.sx_school, cmap=plt.get_cmap('gray'))
        plt.show()

    def sobel_y(self):
        self.sobely_kernel = np.array([[-1,-2,-1],
                                        [0,0,0],
                                        [1,2,1]])
        self.sy_school = self.convolution2d(self.new_school, self.sobely_kernel, 0)
        plt.imshow(self.sy_school, cmap=plt.get_cmap('gray'))
        plt.show()

    def magnitude(self):
        self.mag = (self.sy_school**2 + self.sx_school**2)**(1/2) 
        plt.imshow(self.mag, cmap=plt.get_cmap('gray'))
        plt.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())