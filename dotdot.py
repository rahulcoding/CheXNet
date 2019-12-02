import cv2
import pandas as pd
import os
import numpy 
import io
import cv2
from PIL import Image
import numpy as np
from os import listdir
import skimage.transform
import cv2
import sys
import os
import pickle
from collections import defaultdict
from collections import OrderedDict
import imageio
import skimage
from skimage.io import *
from skimage.transform import *
from collections import OrderedDict

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation
import matplotlib.patches as patches



def dot(image_bytes,prediction):
    '''
    img_folder_path='./00000002_000.png'
    frame=cv2.imread(img_folder_path)'''
    image = Image.open(io.BytesIO(image_bytes))
    frame = np.array(image)
    
    i = prediction[0][4]
    j = i.split(' ')[1:]
    k = numpy.array(j)
    l = k.astype(float)
    x1=int(l[0])
    y1=int(l[1])
    x2=int(l[2])
    y2=int(l[3])
        
        
    a = np.random.randint(9,size=1)
    b = np.random.randint(9,size=1)
    k = b*10 +a
    k = int(k)
    k = str(k)    
    ln = './static/chhati'+k+'.jpg'    

    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),3) #rgb 220,20,60
    #cv2.rectangle(frame,(x1,y1),(x2,y2),(60,20,220),3)
    #print(frame)
    
    cv2.imwrite(ln,frame)
    #cv2.imshow('image',frame)
    #cv2.waitKey(10000)
#    cv2.destroyAllWindows()


    return ln


    







