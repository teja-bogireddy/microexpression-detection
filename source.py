import theano
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing
from keras import backend as K

K.set_image_dim_ordering('th')
img_rows,img_cols,img_depth=15,15,60
X_tr=[]
i = 1
listing1 = os.listdir('Dataset/Anger')
for vid1 in listing1:
    vid1 = 'Dataset/Anger/'+vid1
    frames = []
    cap = cv2.VideoCapture(vid1)
    fps = cap.get(5)
    for k in xrange(60):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    X_tr.append(ipt)
    print i
    i = i + 1
print "Data Loaded"
i = 1
listing2 = os.listdir('Dataset/Disgust')
for vid2 in listing2:
    vid2 = 'Dataset/Disgust/'+vid2
    frames = []
    cap = cv2.VideoCapture(vid2)
    fps = cap.get(5)
    for k in xrange(60):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    X_tr.append(ipt)
    print i
    i = i + 1
print "Data Loaded"
i = 1
listing3 = os.listdir('Dataset/Happy')
for vid3 in listing3:
    vid3 = 'Dataset/Happy/'+vid3
    frames = []
    cap = cv2.VideoCapture(vid3)
    fps = cap.get(5)
    for k in xrange(60):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    X_tr.append(ipt)
    print i
    i = i + 1
print "Data Loaded"

X1_tr = []
i = 1
listing4 = os.listdir('Testset/Angry')
for vid3 in listing4:
    vid3 = 'Testset/Angry/'+vid3
    frames = []
    cap = cv2.VideoCapture(vid3)
    fps = cap.get(5)
    for k in xrange(60):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    X1_tr.append(ipt)
    print i
    i = i + 1
print "Data Loaded"

i = 1
listing4 = os.listdir('Testset/Disgust')
for vid3 in listing4:
    vid3 = 'Testset/Disgust/'+vid3
    frames = []
    cap = cv2.VideoCapture(vid3)
    fps = cap.get(5)
    for k in xrange(60):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    X1_tr.append(ipt)
    print i
    i = i + 1
print "Data Loaded"

i = 1
listing4 = os.listdir('Testset/Happy')
for vid3 in listing4:
    vid3 = 'Testset/Happy/'+vid3
    frames = []
    cap = cv2.VideoCapture(vid3)
    fps = cap.get(5)
    for k in xrange(60):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    X1_tr.append(ipt)
    print i
    i = i + 1
print "Data Loaded"

X1_tr_array = np.array(X1_tr)
X_tr_array = np.array(X_tr)
num1_samples = len(X1_tr_array)
num_samples = len(X_tr_array)

label=np.ones((num_samples,),dtype = int)
label1=np.ones((num1_samples,),dtype = int)
label[0:81]= 0
label[81:162] = 1
label[162:243] = 2
label1[0:23] = 0
label1[23:46] = 1
label1[46:94] = 2


train_data = [X_tr_array,label]
train1_data = [X1_tr_array,label1]
(X_train, y_train) = (train_data[0],train_data[1])
(X1_train, y1_train) = (train1_data[0],train1_data[1])
train_set = np.zeros((num_samples, 1, img_rows, img_cols, img_depth))
train1_set = np.zeros((num1_samples, 1, img_rows, img_cols, img_depth))
for h in xrange(num_samples):
    train_set[h][0][:][:][:]=X_train[h,:,:,:]

for h in xrange(num1_samples):
    train1_set[h][0][:][:][:]=X1_train[h,:,:,:]

batch_size = 20
nb_classes = 3
nb_epoch = 200
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y1_train = np_utils.to_categorical(y1_train, nb_classes)
nb_filters = [32, 32]
nb_pool = [3, 3]
nb_conv = [5, 5]
train_set = train_set.astype('float32')
train_set-= np.mean(train_set)
train_set/= np.max(train_set) 
train1_set = train1_set.astype('float32')
train1_set-= np.mean(train1_set)
train1_set/= np.max(train1_set) 
print "Complete Data is Loaded"

model = Sequential()
model.add(Conv3D(32, (5,5,5), input_shape=(1, img_rows, img_cols, img_depth), activation='relu'))
model.add(MaxPooling3D(pool_size=(3,3,3)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes,kernel_initializer='normal'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='RMSprop',metrics=['accuracy'])
print "Model Is Created"

#X_train_new, X_val_new, y_train_new, y_val_new =  train_test_split(train_set, Y_train, test_size=0.2, random_state=42)
#print "Data Is Split"

hist = model.fit(train_set, Y_train, validation_data=(train1_set,Y1_train),batch_size=batch_size,epochs = nb_epoch, verbose=1)


print "Predicting"
out2 = model.predict(train_set)
print(np.argmax(out2, axis=1))

print "Predicting Again"
out3 = model.predict(train1_set)
print(np.argmax(out3, axis=1))
