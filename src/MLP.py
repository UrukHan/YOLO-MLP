import pandas as pd
import glob
from src.config import CONFIGURATION as cfg
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import pickle
import os
import shutil

import tensorflow as tf
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class MODEL():
    ''' Neural Network Model Class (BERT) '''
    
    def __init__(self):
        super().__init__()

        try:
            self.model = tf.keras.models.load_model(cfg.MODEL)
        except:
            self.model = self.MLPMixer(cfg.PATCH_SIZE_MLP, cfg.PB_MLP, cfg.DS_MLP, cfg.DC_MLP, cfg.NB_MLP, cfg.IMAGE_SIZE_MLP, cfg.BATCH_SIZE_MLP, cfg.OUT_MLP)
            self.loss = tf.keras.losses.BinaryCrossentropy()
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
            self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics='accuracy')

    class MLPBlock(tf.keras.layers.Layer):
        def __init__(self, S, C, DS, DC):
            super().__init__()
            self.layerNorm1 = tf.keras.layers.LayerNormalization()
            self.layerNorm2 = tf.keras.layers.LayerNormalization()
            w_init = tf.random_normal_initializer()
            self.DS = DS
            self.DC = DC
            self.W1 = tf.Variable(
                initial_value=w_init(shape=(S, DS), dtype="float32"),
                trainable=True,
            )
            self.W2 = tf.Variable(
                initial_value=w_init(shape=(DS, S), dtype="float32"),
                trainable=True,
            )
            self.W3 = tf.Variable(
                initial_value=w_init(shape=(C, DC), dtype="float32"),
                trainable=True,
            )
            self.W4 = tf.Variable(
                initial_value=w_init(shape=(DC, C), dtype="float32"),
                trainable=True,
            )

        def call(self, X):
            batch_size, S, C = X.shape
            X_T = tf.transpose(self.layerNorm1(X), perm=(0, 2, 1))
            W1X = tf.matmul(X_T, self.W1) 
            U = tf.transpose(tf.matmul(tf.nn.gelu(W1X), self.W2), perm=(0, 2, 1)) + X
            W3U = tf.matmul(self.layerNorm2(U), self.W3) 
            Y = tf.matmul(tf.nn.gelu(W3U), self.W4) + U  
            return Y


    class MLPMixer(tf.keras.models.Model):
        def __init__(self, patch_size, C, DS, DC, num_of_mlp_blocks, image_size, batch_size, num_classes):
            super().__init__()
            self.S = (image_size * image_size) // (patch_size * patch_size)
            self.projection = tf.keras.layers.Dense(C)
            self.mlpBlocks = [MODEL.MLPBlock(self.S, C, DS, DC) for _ in range(num_of_mlp_blocks)]
            self.batch_size = batch_size
            self.patch_size = patch_size

            self.C = C
            self.DS = DS
            self.DC = DC
            self.image_size = image_size
            self.num_classes = num_classes

            self.classificationLayer = tf.keras.models.Sequential([
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(2048, activation = "leaky_relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(2048, activation = "leaky_relu"),
                tf.keras.layers.Dense(2, activation='softmax')
            ])


        def extract_patches(self, images, patch_size):
            batch_size = tf.shape(images)[0]

            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size, patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patches = tf.reshape(patches, [batch_size, -1, 3 * patch_size ** 2])
            return patches

        def call(self, images):
            batch_size = images.shape[0]
            X = self.extract_patches(images, self.patch_size) 
            X = self.projection(X)
            for block in self.mlpBlocks:
                X = block(X)
            out = self.classificationLayer(X)
            return out


    def train(self):
        shutil.rmtree(cfg.MODEL)
        os.mkdir(cfg.MODEL)
        '''faces = sorted(glob.glob(cfg.DATA + cfg.MLP_TRAIN_DATA + '/*.jpg'))
        df = pd.DataFrame({'paths': faces, 'label': [int(x.split('.')[0][-1]) for x in faces]})
        df.label = df.label - 1
        images = []
        for i in tqdm(df.paths.values):
            img = cv2.imread(i)
            img = cv2.resize(img, (cfg.IMAGE_SIZE_MLP, cfg.IMAGE_SIZE_MLP), interpolation=cv2.INTER_LINEAR)
            images.append(img.astype(int))
        df['images'] = images

        X = np.array([x for x in df.images.values])
        tmp = df.label.values.astype(int)
        y = np.zeros((tmp.size, tmp.max() + 1))
        y[np.arange(tmp.size), tmp] = 1
        y = y.astype(int)
        
        # early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', mode = 'max', patience = 32)
        history = self.model.fit(X, y, validation_split = 0.1,
                            batch_size = cfg.BATCH_SIZE_MLP, epochs = cfg.EPOHS_MLP, 
                            #callbacks=[early_stop], 
                            verbose = 1)
        '''

        train_dir = cfg.DATA + cfg.MLP_TRAIN_DATA 

        image_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1/255,
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=(0.2, 2),
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.1)


        train_generator = image_gen.flow_from_directory(
                            directory=train_dir,
                            batch_size=cfg.BATCH_SIZE_MLP,                                                        
                            shuffle=True,
                            target_size=(cfg.IMAGE_SIZE_MLP,cfg.IMAGE_SIZE_MLP),
                            subset='training')

        validation_generator = image_gen.flow_from_directory(
                            directory=train_dir,
                            batch_size=cfg.BATCH_SIZE_MLP,                                                        
                            shuffle=True,
                            target_size=(cfg.IMAGE_SIZE_MLP,cfg.IMAGE_SIZE_MLP),
                            subset='validation')

        history = self.model.fit_generator(
                            train_generator,
                            steps_per_epoch=int(np.ceil(train_generator.samples / float(cfg.BATCH_SIZE_MLP))),
                            validation_data = validation_generator,
                            validation_steps = int(np.ceil(validation_generator.n / float(cfg.BATCH_SIZE_MLP))),
                            epochs = cfg.EPOHS_MLP, 
                            verbose = 1)

        '''with open('data/historyDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(cfg.DATA + cfg.PLT)

        self.model.save(cfg.MODEL)'''

    def predict(self):
        faces = sorted(glob.glob(cfg.FACES + '/*.jpg'))
        with open(cfg.DATA + cfg.DICT_CLASS, 'rb') as f:
            dict_class = pickle.load(f)
        n = 0
        for i in faces:
            img = cv2.imread(i)
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
            X = np.array([img.astype(int)])
            pred = self.model.predict(X)[0][1]
            print(pred)
            if pred > cfg.PRED_MLP_TRH:
                dict_class[i.split('_')[0].split('\\')[-1]] = 1
                cv2.imwrite(cfg.DATA + cfg.CHILD + i.split('_')[0].split('\\')[-1] + '_' + str(n) + ".jpg", img)
                n += 1
            else:
                try:
                    if dict_class[i.split('_')[0].split('\\')[-1]] != 1:
                        dict_class[i.split('_')[0].split('\\')[-1]] = 0
                except:
                    dict_class[i.split('_')[0].split('\\')[-1]] = 0
                cv2.imwrite(cfg.DATA + cfg.ADULT + i.split('_')[0].split('\\')[-1] + '_' + str(n) + ".jpg", img)
                n += 1
        with open(cfg.DATA + cfg.DICT_CLASS, 'wb') as f:
            pickle.dump(dict_class, f)
        return dict_class



