from joblib import Parallel, delayed
import os
import numpy as np
import multiprocessing
from PIL import Image
from tqdm import tqdm, trange

class DataImporter(object):
    
    @staticmethod
    def img_data(path,img_width, img_height, img_channels):
        def load_img(image):
                img = Image.open(path +'/'+ image)
                arr = np.asarray(img, dtype = "float32")
                assert arr.shape[0] == img_width and arr.shape[1] == img_height and arr.shape[2] == img_channels, 'the data image size is different from setting image size'
                data[i,:,:,:] = arr
        images = os.listdir(path)
        data_size = len(images)
        data = np.empty((data_size, img_width, img_height,img_channels),dtype = 'float32')
        for i in trange(data_size, desc='loading data in "{}"'.format(path)):
            load_img(images[i])
        return data


print(multiprocessing.cpu_count())          
            
data_path = '../Data/faces'
data = DataImporter.img_data(data_path,96,96,3)

        



