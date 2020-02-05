#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras
print("TensorFlow version is ", tf.__version__)

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[2]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

import os, os.path, shutil

folder_path = "TResistors"
target_path = 'TransferLearning'

images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for image in images:
    folder_name = image.split(' (')[0]

    new_path = os.path.join(folder_path, folder_name)
    t_path = os.path.join(target_path, folder_name)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    if not os.path.exists(t_path):
        os.makedirs(t_path)
        

    old_image_path = os.path.join(folder_path, image)
    new_image_path = os.path.join(new_path, image)
    shutil.move(old_image_path, new_image_path)
    img = load_img(new_image_path)
    print("images: ",new_image_path)
    x = img_to_array(img)  
    x = x.reshape((1,) + x.shape)
    save_dir = target_path +'/'+ folder_name
    
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_prefix=folder_name, 
                              save_to_dir=save_dir, save_format='png'):
        print("the current save dir: ",save_dir)
        print("accessing f_name: \n", folder_name)
        
        i += 1
        if i > 30:
            break


# In[3]:


base_dir, _ = os.path.splitext('TransferLearning')


# In[4]:


train_dir = os.path.join(base_dir, 'TrainClass')
validation_dir = os.path.join(base_dir, 'ValidateClass')

# Directory with our training ClassA pictures
train_ClassA_dir = os.path.join(train_dir, 'ClassA')
print ('Total training ClassA images:', len(os.listdir(train_ClassA_dir)))

# Directory with our training ClassB pictures
train_ClassB_dir = os.path.join(train_dir, 'ClassB')
print ('Total training ClassB images:', len(os.listdir(train_ClassB_dir)))

# Directory with our training ClassC pictures
train_ClassC_dir = os.path.join(train_dir, 'ClassC')
print ('Total training ClassC images:', len(os.listdir(train_ClassC_dir)))

# Directory with our training ClassD pictures
train_ClassD_dir = os.path.join(train_dir, 'ClassD')
print ('Total training ClassD images:', len(os.listdir(train_ClassD_dir)))

# Directory with our training ClassE pictures
train_ClassE_dir = os.path.join(train_dir, 'ClassE')
print ('Total training ClassE images:', len(os.listdir(train_ClassE_dir)))

# Directory with our training ClassF pictures
train_ClassF_dir = os.path.join(train_dir, 'ClassF')
print ('Total training ClassF images:', len(os.listdir(train_ClassF_dir)))

# Directory with our training ClassG pictures
train_ClassG_dir = os.path.join(train_dir, 'ClassG')
print ('Total training ClassG images:', len(os.listdir(train_ClassG_dir)))

# Directory with our training ClassH pictures
train_ClassH_dir = os.path.join(train_dir, 'ClassH')
print ('Total training ClassH images:', len(os.listdir(train_ClassH_dir)))

# Directory with our training ClassI pictures
train_ClassI_dir = os.path.join(train_dir, 'ClassI')
print ('Total training ClassI images:', len(os.listdir(train_ClassI_dir)))

# Directory with our training ClassJ pictures
train_ClassJ_dir = os.path.join(train_dir, 'ClassJ')
print ('Total training ClassJ images:', len(os.listdir(train_ClassJ_dir)))

# Directory with our training ClassK pictures
train_ClassK_dir = os.path.join(train_dir, 'ClassK')
print ('Total training ClassK images:', len(os.listdir(train_ClassK_dir)))

# Directory with our training ClassL pictures
train_ClassL_dir = os.path.join(train_dir, 'ClassL')
print ('Total training ClassL images:', len(os.listdir(train_ClassL_dir)))

# Directory with our validation ClassA pictures
validation_ClassA_dir = os.path.join(validation_dir, 'ClassA')
print ('Total validation ClassA images:', len(os.listdir(validation_ClassA_dir)))

validation_ClassB_dir = os.path.join(validation_dir, 'ClassB')
print ('Total validation ClassB images:', len(os.listdir(validation_ClassB_dir)))

validation_ClassC_dir = os.path.join(validation_dir, 'ClassC')
print ('Total validation ClassC images:', len(os.listdir(validation_ClassC_dir)))

validation_ClassD_dir = os.path.join(validation_dir, 'ClassD')
print ('Total validation ClassD images:', len(os.listdir(validation_ClassD_dir)))

validation_ClassE_dir = os.path.join(validation_dir, 'ClassE')
print ('Total validation ClassE images:', len(os.listdir(validation_ClassE_dir)))

validation_ClassF_dir = os.path.join(validation_dir, 'ClassF')
print ('Total validation ClassF images:', len(os.listdir(validation_ClassF_dir)))

validation_ClassG_dir = os.path.join(validation_dir, 'ClassG')
print ('Total validation ClassG images:', len(os.listdir(validation_ClassG_dir)))

validation_ClassH_dir = os.path.join(validation_dir, 'ClassH')
print ('Total validation ClassH images:', len(os.listdir(validation_ClassH_dir)))

validation_ClassI_dir = os.path.join(validation_dir, 'ClassI')
print ('Total validation ClassI images:', len(os.listdir(validation_ClassI_dir)))

validation_ClassJ_dir = os.path.join(validation_dir, 'ClassJ')
print ('Total validation ClassJ images:', len(os.listdir(validation_ClassJ_dir)))

validation_ClassK_dir = os.path.join(validation_dir, 'ClassK')
print ('Total validation ClassK images:', len(os.listdir(validation_ClassK_dir)))

validation_ClassL_dir = os.path.join(validation_dir, 'ClassL')
print ('Total validation ClassL images:', len(os.listdir(validation_ClassL_dir)))

# Directory with our validation ClassB pictures
#validation_ClassB_dir = os.path.join(validation_dir, 'ClassB')
#print ('Total validation ClassB images:', len(os.listdir(validation_ClassB_dir)))


# In[ ]:





# In[5]:


image_size = 160 # All images will be resized to 160x160
batch_size = 32

# Rescale all images by 1./255 and apply image augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255)

validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
                train_dir,  # Source directory for the training images
                target_size=(image_size, image_size),  
                batch_size=batch_size,
                # Since we use binary_crossentropy loss, we need binary labels
                class_mode='binary')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(
                validation_dir, # Source directory for the validation images
                target_size=(image_size, image_size),
                batch_size=batch_size,
                class_mode='binary')


# In[6]:


IMG_SHAPE = (image_size, image_size, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False, 
                                               weights='imagenet')


# In[7]:


base_model.trainable = False


# In[8]:


# Let's take a look at the base model architecture
base_model.summary()


# In[10]:


model = tf.keras.Sequential([
  base_model,
  keras.layers.GlobalAveragePooling2D(),
  keras.layers.Dense(1, activation='sigmoid')
])


# In[11]:


model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])


# In[12]:


model.summary()


# In[13]:


len(model.trainable_variables)


# In[ ]:


epochs = 10
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

history = model.fit_generator(train_generator, 
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs, 
                              workers=4,
                              validation_data=validation_generator, 
                              validation_steps=validation_steps)


# In[ ]:


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()


# In[ ]:


base_model.trainable = True


# In[ ]:


# how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False


# In[ ]:


model.compile(loss='binary_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


len(model.trainable_variables)


# In[ ]:


history_fine = model.fit_generator(train_generator, 
                                   steps_per_epoch = steps_per_epoch,
                                   epochs=epochs, 
                                   workers=4,
                                   validation_data=validation_generator, 
                                   validation_steps=validation_steps)


# In[ ]:


accuracy += history_fine.history['accuracy']
val_accuracy += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']


# In[ ]:


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.ylim([0.9, 1])
plt.plot([epochs-1,epochs-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 0.2])
plt.plot([epochs-1,epochs-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

