import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'test-dataset:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F399530%2F767109%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240515%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240515T173914Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D57f9f97f670bba5caf101e0eeb7ab2dd305ba8deadf8d6b3a3d4e10ad596ec21d6259b45106f2ff485fa55e8b305c9b57be420fd20a41bb2e9ed242fb4debec8e24a4286219b3b6c9d6cd3424220d0fa99176efc0c84ea84414f753a2abe773e402bbd064e2839c2183b564e0dfac194a0c5f85ea33b79a1726d482ab32f7747bb2be640f0fedc5a8696df3424c3bd8b560b888818dcd67735d374df106295d485a9926e9b353d140697ddcbf38affb51099f80e6e1ae81f69c724c52ff66cbc413d53006113bf3f427bad0a59438d8476da4fe7058a57653be29b2c183eb9e56360c456d2e6f0c981bee5178cc79de8e98e1da0f70b6ab14d47a2dbbceccda2'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

# Ignore  the warnings
import warnings

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
%matplotlib
inline
style.use('fivethirtyeight')
sns.set(style='whitegrid', color_codes=True)

# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# preprocess.
from keras.preprocessing.image import ImageDataGenerator

# dl libraraies
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint

# specifically for cnn
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import InputLayer

import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2
import numpy as np
from tqdm import tqdm
import os
from random import shuffle
from zipfile import ZipFile
from PIL import Image

from keras.applications.resnet50 import ResNet50
import pathlib
import re
import shutil
from scipy import ndarray
import random
import skimage as sk
from skimage import io
from skimage.util import img_as_ubyte
from sklearn.metrics import classification_report

IMG_SIZE = 256
data_dir = '/kaggle/input/test-dataset/Fire-Detection'
data_dir = pathlib.Path(data_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(class_names)

image_count = len(list(data_dir.glob('*/*.jpg'))) + len(list(data_dir.glob('*/*.JPG'))) + len(list(data_dir.glob('*/*.png')))  + len(list(data_dir.glob('*/*.jpeg')))
print(image_count)

# Plot histogram
f = [item_num for item_num in data_dir.rglob('*')]
values, counts = np.unique([x.parent for x in f ], return_counts=True)
#print(list(zip(counts, values)))
x_name = tuple(class_names)
y_pos = np.arange(len(x_name))
x_value = list(counts)
print(x_value)
x_value.pop(0)
print(x_value)
plt.subplots(figsize=(12,6))
bars = plt.bar(y_pos, x_value, align='center', alpha=1)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 2, yval, ha='center', va='bottom')
plt.xticks(y_pos, x_name)
plt.xlabel('Label')
plt.ylabel('Number of files')
plt.title('Data')
plt.show()

for i in range(0, 2):
  var = pathlib.Path("/kaggle/input/test-dataset/Fire-Detection/"+str(i)+"/*")
  list_ds = tf.data.Dataset.list_files(str(var), shuffle=True)
  image_count = len(list(data_dir.glob(str(i)+'/*.jpg'))) + len(list(data_dir.glob(str(i)+'/*.JPG'))) + len(list(data_dir.glob(str(i)+'/*.png')))  + len(list(data_dir.glob(str(i)+'/*.jpeg')))
  val_size = int(image_count * 0.3) #0.3
  train_ds_list = list_ds.skip(val_size)
  globals()['train_ds_list_%s' % i] = train_ds_list
  list_a = list_ds.take(val_size) #get first 30% images from list_ds
  list_a_size = tf.data.experimental.cardinality(list_a).numpy()
  test_ds_list = list_a.skip(list_a_size * 0.5)
  globals()['test_ds_list_%s' % i] = test_ds_list
  val_ds_list = list_a.take(list_a_size * 0.5)
  globals()['val_ds_list_%s' % i] = val_ds_list

val_ds_list = val_ds_list_0.concatenate(val_ds_list_1)
test_ds_list = test_ds_list_0.concatenate(test_ds_list_1)
train_ds_list = train_ds_list_0.concatenate(train_ds_list_1)

data_dir_train = pathlib.Path("/kaggle/working/Fire-Detection-train")
if not os.path.exists(data_dir_train):
  os.makedirs(data_dir_train)

#paste above train_ds_list to data_dir_train
for f in train_ds_list.as_numpy_iterator():
  link = f.decode("utf-8")
  substring = re.search('(.*)/(.*)/(.*)', link)
  dst = pathlib.Path("/kaggle/working/Fire-Detection-train/"+substring.group(2))
  if not os.path.exists(dst):
    os.makedirs(dst)
  shutil.copy2(link, dst)

# Plot histogram for dara_dir_train
f2 = [item_num2 for item_num2 in data_dir_train.rglob('*')]
values2, counts2 = np.unique([x2.parent for x2 in f2 ], return_counts=True)
x_value2 = list(counts2)
#print(list(zip(counts2, values2)))
x_name2 = tuple(class_names)
y_pos2 = np.arange(len(x_name2))
x_value2.pop(0)
print(x_value2)
plt.subplots(figsize=(12,6))
bars = plt.bar(y_pos2, x_value2, align='center', alpha=1)
for bar in bars:
    yval2 = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval2 + 2, yval2, ha='center', va='bottom')
plt.xticks(y_pos2, x_name2)
plt.xlabel('Label')
plt.ylabel('Number of files')
plt.title('Data')
plt.show()

data_dir_val = pathlib.Path("/kaggle/working/Fire-Detection-val")
if not os.path.exists(data_dir_val):
  os.makedirs(data_dir_val)

#paste above val_ds_list to data_dir_val
for f in val_ds_list.as_numpy_iterator():
  link = f.decode("utf-8")
  substring = re.search('(.*)/(.*)/(.*)', link)
  dst = pathlib.Path("/kaggle/working/Fire-Detection-val/"+substring.group(2))
  if not os.path.exists(dst):
    os.makedirs(dst)
  shutil.copy2(link, dst)

# Plot histogram for data_dir_val
f2 = [item_num2 for item_num2 in data_dir_val.rglob('*')]
values2, counts2 = np.unique([x2.parent for x2 in f2 ], return_counts=True)
x_value2 = list(counts2)
#print(list(zip(counts2, values2)))
x_name2 = tuple(class_names)
y_pos2 = np.arange(len(x_name2))
x_value2.pop(0)
print(x_value2)
plt.subplots(figsize=(12,6))
bars = plt.bar(y_pos2, x_value2, align='center', alpha=1)
for bar in bars:
    yval2 = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval2 + 2, yval2, ha='center', va='bottom')
plt.xticks(y_pos2, x_name2)
plt.xlabel('Label')
plt.ylabel('Number of files')
plt.title('Data')
plt.show()

data_dir_test = pathlib.Path("/kaggle/working/Fire-Detection-test")
if not os.path.exists(data_dir_test):
  os.makedirs(data_dir_test)

#paste above test_ds_list to data_dir_test
for f in test_ds_list.as_numpy_iterator():
  link = f.decode("utf-8")
  substring = re.search('(.*)/(.*)/(.*)', link)
  dst = pathlib.Path("/kaggle/working/Fire-Detection-test/"+substring.group(2))
  if not os.path.exists(dst):
    os.makedirs(dst)
  shutil.copy2(link, dst)


  def none(filename):
      return filename


  def histogram(filename):
      colorimage_b = cv2.equalizeHist(filename[:, :, 0])
      colorimage_g = cv2.equalizeHist(filename[:, :, 1])
      colorimage_r = cv2.equalizeHist(filename[:, :, 2])
      colorimage_e = np.stack((colorimage_b, colorimage_g, colorimage_r), axis=2)
      return colorimage_e


  # dictionary of the transformations we defined earlier
  available_transformations = {
      # 'none': none,
      'histogram': histogram
  }

  for i in range(0, 2):
      num_files_desired = 600
      folder_path = os.path.join(str(data_dir_train) + "/" + str(i))
      image_count = len(list(data_dir_train.glob(str(i) + '/*.jpg'))) + len(
          list(data_dir_train.glob(str(i) + '/*.JPG'))) + len(list(data_dir_train.glob(str(i) + '/*.png'))) + len(
          list(data_dir_train.glob(str(i) + '/*.jpeg')))
      plus = num_files_desired - image_count - 1

      if plus >= 0:
          images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                    os.path.isfile(os.path.join(folder_path, f))]
          num_generated_files = 0
          while num_generated_files <= plus:
              image_path = random.choice(images)  # random image from the folder
              image_to_transform = sk.io.imread(image_path)
              num_transformations_to_apply = 1
              num_transformations = 0
              transformed_image = None
              while num_transformations <= num_transformations_to_apply:
                  # random transformation to apply for a single image
                  key = random.choice(list(available_transformations))
                  transformed_image = available_transformations[key](image_to_transform)
                  transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGBA2BGR)
                  transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
                  num_transformations += 1
                  new_file_path = '%s/augmented_%s.JPG' % (folder_path, num_generated_files)
                  io.imsave(new_file_path, img_as_ubyte(transformed_image))
              num_generated_files += 1


# Plot histogram for dara_dir_train after enhance data
f2 = [item_num2 for item_num2 in data_dir_train.rglob('*')]
values2, counts2 = np.unique([x2.parent for x2 in f2 ], return_counts=True)
x_value2 = list(counts2)
#print(list(zip(counts2, values2)))
x_name2 = tuple(class_names)
y_pos2 = np.arange(len(x_name2))
x_value2.pop(0)
print(x_value2)
plt.subplots(figsize=(12,6))
bars = plt.bar(y_pos2, x_value2, align='center', alpha=1)
for bar in bars:
    yval2 = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval2 + 2, yval2, ha='center', va='bottom')
plt.xticks(y_pos2, x_name2)
plt.xlabel('Label')
plt.ylabel('Number of files')
plt.title('Data')
plt.show()


def make_train_data(label, DIR):
    for img in tqdm(os.listdir(DIR)):
        path = os.path.join(DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X_train.append(np.array(img))
        Z_train.append(str(label))


X_train = []
Z_train = []

NOTFIRE = '/kaggle/working/Fire-Detection-train/0'
FIRE = '/kaggle/working/Fire-Detection-train/1'

make_train_data('NOTFIRE', NOTFIRE)
make_train_data('FIRE', FIRE)

np.shape(X_train)
np.save("/kaggle/working/X_train.npy", X_train)

le = LabelEncoder()
Y_train = le.fit_transform(Z_train)
Y_train = to_categorical(Y_train, 2)
# print(Y_train)

X_train = np.load("/kaggle/working/X_train.npy")
X_train = np.array(X_train)
# x_train,x_val,y_train,y_val=train_test_split(X_train,Y_train,test_size=0.3,random_state=42)

def make_val_data(label, DIR):
    for img in tqdm(os.listdir(DIR)):
        path = os.path.join(DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X_val.append(np.array(img))
        Z_val.append(str(label))


X_val = []
Z_val = []

NOTFIRE = '/kaggle/working/Fire-Detection-val/0'
FIRE = '/kaggle/working/Fire-Detection-val/1'

make_val_data('NOTFIRE', NOTFIRE)
make_val_data('FIRE', FIRE)

np.shape(X_val)
np.save("/kaggle/working/X_val.npy", X_val)

le = LabelEncoder()
Y_val = le.fit_transform(Z_val)
Y_val = to_categorical(Y_val, 2)
# print(Y_val)

X_val = np.load("/kaggle/working/X_val.npy")
X_val = np.array(X_val)


def make_test_data(label, DIR):
    for img in tqdm(os.listdir(DIR)):
        path = os.path.join(DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X_test.append(np.array(img))
        Z_test.append(str(label))


X_test = []
Z_test = []

NOTFIRE = '/kaggle/working/Fire-Detection-test/0'
FIRE = '/kaggle/working/Fire-Detection-test/1'

make_test_data('NOTFIRE', NOTFIRE)
make_test_data('FIRE', FIRE)

np.shape(X_test)
np.save("/kaggle/working/X_test.npy", X_test)

le = LabelEncoder()
Y_test = le.fit_transform(Z_test)
Y_test = to_categorical(Y_test, 2)
# print(Y_test)

X_test = np.load("/kaggle/working/X_test.npy")
X_test = np.array(X_test)

fig,ax=plt.subplots(2,5)
plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)
fig.set_size_inches(10,10)

for i in range(2):
    for j in range (5):
        l=rn.randint(0,len(Z_train))
        ax[i,j].grid(False)
        ax[i,j].imshow(X_train[l][:,:,::-1])
        ax[i,j].set_title(Z_train[l])
        ax[i,j].set_aspect('equal')

base_model=ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_SIZE,IMG_SIZE,3), pooling='max')
#base_model=ResNet50(include_top=False, weights=None,input_shape=(224,224,3), pooling='max')
#base_model.summary()

model=Sequential()
model.add(base_model)
model.add(Dense(2048,activation='relu'))
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(2,activation='softmax'))

epochs=20
batch_size=64

base_model.trainable=False

model.compile(optimizer=Adam(lr=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

History = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val,Y_val))

plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

y_predict=model.predict(X_test)
y_predict=np.argmax(y_predict, axis=1)
y_true=np.argmax(Y_test, axis=1)
print(classification_report(y_true, y_predict, labels=[0,1]))