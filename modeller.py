from __future__ import print_function

import cv2
import math
import numpy as np
import os

import keras
from keras.models import Sequential, load_model
from keras.layers import Cropping2D, Lambda
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import matplotlib.image as mpim

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def get_samples_list(sampledir, exclude=[], d_range=[]):
    samples = []
    labelcount = []
    for fname in os.listdir(sampledir):
        if fname[-4:]=='.png' and not fname[0] in exclude:
            d = int(fname.split('__')[-1].split('_')[0])
            if len(d_range) > 0 and (d < d_range[0] or d > d_range[1]):
                continue
            label = int(fname[0])
            samples.append((os.path.join(sampledir, fname), label))
            while len(labelcount) < label+1:
                labelcount += [0]
            labelcount[label] += 1
    print('label counts:')
    for i in range(len(labelcount)):
        print('{}: {}'.format(i, labelcount[i]))
    print('total: {}'.format(len(samples)))
    return samples


def augment_samples_list(samples, mult=[1, 1, 1], tx=[0, 0]):
    augs = []
    for sample in samples:
        x = mult[sample[1]]-1
        for i in range(x):
            shiftx = np.random.randint(tx[0], tx[1])
            shifty = np.random.randint(tx[0], tx[1])
            sample_aug = [sample[0], sample[1], shiftx, shifty]
            augs.append(sample_aug)
    return samples + augs


def datagen(samples, batch_size=32, n_class=4):
    n_samples = len(samples)
    while True:
        samples = shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            labels = []
            for batch_sample in batch_samples:
                im = mpim.imread(batch_sample[0])
                if len(batch_sample) > 2:
                    tx = batch_sample[2]
                    ty = batch_sample[3]
                    nrow, ncol = im.shape[:2]
                    T = np.float32([[1, 0, tx], [0, 1, ty]])
                    im = cv2.warpAffine(im, T, (ncol, nrow))
                images.append(im)
                labels.append(batch_sample[1])

            images = np.array(images)
            labels = keras.utils.to_categorical(np.array(labels), num_classes=n_class)
            yield shuffle(images, labels)


def count_sample_distro(samples):
    label_counts = []
    for sample in samples:
        label = sample[1]
        while len(label_counts) < label+1:
            label_counts += [0]
        label_counts[label] += 1

    total_count = sum(label_counts)
    max_count = max(label_counts)
    print('label counts, proportion, mult')
    for i, label in enumerate(label_counts):
        print('{} {:5.3f} {:6.2f}'.format(label, 1.0*label/total_count, 1.0*max_count/label))
    print('total', total_count)
    return label_counts


def get_samples_list_recursive(sample_root, exclude=[]):
    fnames = os.listdir(sample_root)
    samples = []
    for fname in fnames:
        if not (fname[0] in exclude) and fname[-4:] == '.png':
            samples.append([os.path.join(sample_root, fname), int(fname[0])])
        elif os.path.isdir(os.path.join(sample_root, fname)):
            new_root = os.path.join(sample_root, fname)
            add_samples = get_samples_list_recursive(new_root, exclude)
            samples += add_samples
    return samples


def train(model, sample_dir='samples', batch_size=8, epochs=1):
    samples = get_samples_list_recursive(sample_dir, exclude=['3'])
    samples = augment_samples_list(samples, mult=[1, 20, 6], tx=[-20, 21])
    _ = count_sample_distro(samples)

    train_samples, valid_samples = train_test_split(samples, test_size=0.25)
    train_gen = datagen(train_samples, batch_size=batch_size, n_class=3)
    valid_gen = datagen(valid_samples, batch_size=batch_size, n_class=3)

    train_step = math.ceil(len(train_samples)/batch_size)
    valid_step = math.ceil(len(valid_samples)/batch_size)

    history = model.fit_generator(train_gen, steps_per_epoch=train_step,
                                  validation_data=valid_gen, validation_steps=valid_step,
                                  epochs=epochs, verbose=1)
    model.save('model.h5')
    return history


def net_nvidia(n_class=3):
    model = Sequential()

    model.add(Cropping2D(((100, 100), (250, 250)), input_shape=(600, 800, 3)))

    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2)))
    model.add(Activation('relu')) # the paper doesn't mention activation function, but isn't that needed?

    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1)))
    model.add(Dropout(0.30))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(n_class))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-6), metrics=['accuracy'])

    return model


def net_simple(n_class=3):
    model = Sequential()

    model.add(Cropping2D((100, 250), input_shape=(600, 800, 3)))

    model.add(Conv2D(8, kernel_size=(5, 5)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(16, kernel_size=(5, 5)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(32, kernel_size=(5, 5)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, kernel_size=(5, 5)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(n_class))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(lr=1e-6), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def net2(n_class=3):
    model = Sequential()

    model.add(Cropping2D((100, 250), input_shape=(600, 800, 3)))

    model.add(Conv2D(16, kernel_size=(5, 5)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, kernel_size=(5, 5)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=(5, 5)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(16))
    model.add(Activation('relu'))

    model.add(Dense(n_class))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(lr=1e-6), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
