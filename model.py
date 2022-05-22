#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[ ]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
#get_ipython().run_line_magic('matplotlib', 'inline')
import sys
from skimage.feature import hog
import pickle


# In[ ]:



vehicles_dir = sys.argv[1]
non_vehicles_dir = sys.argv[2]

# images are divided up into vehicles and non-vehicles
cars = []
notcars = []

# Read vehicle images
images = glob.iglob(vehicles_dir + '/**/*.png', recursive=True)

for image in images:
    cars.append(image)

# Read non-vehicle images
images = glob.iglob(non_vehicles_dir + '/**/*.png', recursive=True)

for image in images:
    notcars.append(image)

#data_info = data_look(cars, notcars)  # TO SHOW THE DATA OF DATA SET

#print('Your function returned a count of',
#      data_info["n_cars"], ' cars and',
#      data_info["n_notcars"], ' non-cars')
#print('of size: ', data_info["image_shape"], ' and data type:',
#      data_info["data_type"])


# # HOG features

# In[ ]:




def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    """
    Return the hog features of the given input image
    Call with two outputs if vis==True"""
    # THE RETURN HOG FEATURES & HOG IMAGE 
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
     # THE RETURN HOG FEATURES ONLY
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=( cell_per_block, cell_per_block), transform_sqrt=True,
                       visualize=vis, feature_vector=feature_vec)
        return features


# In[ ]:


# Define a function to extract features from a list of image
def extract_features(imgs, cspace='RGB', orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        image = (image * 255).astype(np.uint8)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True FOR each channel 
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)

            # Append the new feature vector to the features list
            file_features.append(hog_features)
        #features.append(file_features) # in all features Lisr
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# # Training

# In[ ]:


import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

colorspace = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # Length of the vector [ 9 angles ]
pix_per_cell = 8  # each cell 8 * 8
cell_per_block = 2  # Block 2*2
t = time.time()

car_features = extract_features(cars, cspace=colorspace, orient=orient,
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block
                                )
notcar_features = extract_features(notcars, cspace=colorspace, orient=orient,
                                   pix_per_cell=pix_per_cell, cell_per_block=cell_per_block
                                   )
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)  # Standarization to the features & Scaling

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

print(len(y))  # THE TARGET [ONES FOR CARS ] , [ZEROS FOR NON CARS ]

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.15, random_state=rand_state)  # sikit learn / open cv [16000 pics ]
#[0.85 trainig] a5tyar aldata random

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC X_scaler
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')


# # Test The Accuracuy

# In[ ]:


print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4)) # 4 decimal after point
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict])) ## TEST 10 IMAGES 
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


# # Save The pickle file

# In[ ]:



#Pickle the data as it takes a lot of time to generate it

# data_file = './svc_pickle.p'
pickle.dump(svc, open('svc_pickle.pkl', "wb"))
# if not os.path.isfile(data_file):
#     with open(data_file, 'wb') as pfile:
#         pickle.dump(
#             {
#                 'svc': svc,
#                 'scaler': X_scaler,
#                 'orient': orient,
#                 'pix_per_cell': pix_per_cell,
#                 'cell_per_block': cell_per_block,
#             },
            # pfile, pickle.HIGHEST_PROTOCOL)

# print('Data saved in pickle file')

