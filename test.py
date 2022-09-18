#%%
 
import requests
from src.config import CONFIGURATION  as cfg
import pandas as pd
import glob
import pickle 
import tensorflow as tf
import os
from tqdm import tqdm

#%%

url = 'http://localhost:8000/health/'
r = requests.get(url) 
r.json

#%%

url = 'http://localhost:8000/parse/'
r = requests.get(url) 
r.json

#%%

#%%

'''dict_class = {}
vid = sorted(glob.glob(cfg.DATA + cfg.VID_PTH + '/*.mp4'))
for i in tqdm(vid):
    tmp = i.split('\\')[-1].split('.')[0]
    dict_class[tmp] = -1

with open(cfg.DATA + cfg.DICT_CLASS, 'wb') as f:
    pickle.dump(dict_class, f)'''
    
#%%

with open(cfg.DATA + cfg.DICT_CLASS, 'rb') as f:
    dict_class = pickle.load(f)
dict_class

#%%

data = pd.read_csv(cfg.DATA + cfg.VID_CSV)

predict = []
for i in [x.split('/')[-1].split('.')[0].replace('_', '-') for x in data.video_uuid]:
    try:
        predict.append(dict_class[i])
    except:
        predict.append(-2)
data['predict'] = predict
data

#%%

vid = sorted(glob.glob('runs/detect/exp/crops/face/' + '*.jpg'))
for i in tqdm(vid):
    os.rename('runs/detect/exp/crops/face/' + i.split('\\')[-1], 'runs/detect/exp/crops/face/' + i.split('\\')[-1].replace('_', '-'))

#%%



#%%

data.loc[(data['class'] == 0) & (data['class'] == data['predict'])].shape[0]

#%%

data.loc[(data['class'] == 1) & (data['predict'] == -1)].shape[0]

#%%


#%%



'''


    258/430     true positive              279/430
    139/430     no faces / positive        139/430
    33/430      false negative             12/430

    52/430     true negative               36/430
    165/430     no faces / positive        165/430
    213/430      false positive            229/430


'''

#%%


#%%

url = 'http://localhost:8000/slice/'
r = requests.get(url)
r.json

#%%

url = 'http://localhost:8000/train_yolo/'
r = requests.get(url)
r.json

#%%

url = 'http://localhost:8000/detect_yolo/'
r = requests.get(url)
r.json

#%%

# MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP

#%%

url = 'http://localhost:8000/train/'
r = requests.get(url)
r.json()


#%%

url = 'http://localhost:8000/predict/'
r = requests.get(url)
r.json()


#%%


#%%

train_dir = cfg.DATA + cfg.MLP_TRAIN_DATA 

image_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(0.2, 2),
    zoom_range=0.2,
    
    horizontal_flip=True,
    validation_split=0.1)


train_data_gen = image_gen.flow_from_directory(
                                                batch_size=cfg.BATCH_SIZE_MLP,
                                                directory=train_dir,
                                                shuffle=True,
                                                target_size=(cfg.IMAGE_SIZE_MLP,cfg.IMAGE_SIZE_MLP)
                                                )


#%%

train_data_gen.samples


# %%
