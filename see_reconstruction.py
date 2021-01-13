import numpy as np 
from keras.models import load_model
import matplotlib.pyplot as plt
from numpy import load
import pickle

#pickle_in = open('reconstruction.pkl', 'rb')
#reconstructor = pickle.load(pickle_in)

features_symbol = load('features_data.npy')
print(features_symbol.shape)
features_symbol_predicted = load('features_predicted.npy')
print(features_symbol_predicted.shape)

plt.imshow(features_symbol[8, :, :, 0], cmap='gray')
plt.show()
plt.imshow(features_symbol_predicted[8, :, :, 0], cmap='gray')
plt.show()

reconstruction = load_model('reconstruction_model')
