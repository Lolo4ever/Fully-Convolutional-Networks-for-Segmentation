#segmented images
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from keras.applications.vgg16 import VGG16

train_imgs = np.load('./data_files/images_training.npy')
train_seg = np.load('./data_files/seg_training.npy')
test_imgs = np.load('./data_files/images_testing.npy')

n_train = train_imgs.shape[0]
n_test = test_imgs.shape[0]

print ("The number of training images : {}, shape : {}".format(n_train, train_imgs.shape))
print ("The number of testing images : {}, shape : {}".format(n_test, test_imgs.shape))



idx = np.random.randint(n_train)

plt.figure(figsize = (16,14))
plt.subplot(3,1,1)
plt.imshow(train_imgs[idx])
plt.axis('off')
plt.subplot(3,1,2)
plt.imshow(train_seg[idx][:,:,0])
plt.axis('off')
plt.subplot(3,1,3)
plt.imshow(train_seg[idx][:,:,1])
plt.axis('off')
plt.show()

#VGG16 model for encoder


model = VGG16(weights = 'imagenet')

model.summary()

vgg16_weights = model.get_weights()

weights = {
    'conv1_1' : tf.constant(vgg16_weights[0]),
    'conv1_2' : tf.constant(vgg16_weights[2]),
    
    'conv2_1' : tf.constant(vgg16_weights[4]),
    'conv2_2' : tf.constant(vgg16_weights[6]),
    
    'conv3_1' : tf.constant(vgg16_weights[8]),
    'conv3_2' : tf.constant(vgg16_weights[10]),
    'conv3_3' : tf.constant(vgg16_weights[12]),
    
    'conv4_1' : tf.constant(vgg16_weights[14]),
    'conv4_2' : tf.constant(vgg16_weights[16]),
    'conv4_3' : tf.constant(vgg16_weights[18]),
    
    'conv5_1' : tf.constant(vgg16_weights[20]),
    'conv5_2' : tf.constant(vgg16_weights[22]),
    'conv5_3' : tf.constant(vgg16_weights[24]),
}

biases = {
    'conv1_1' : tf.constant(vgg16_weights[1]),
    'conv1_2' : tf.constant(vgg16_weights[3]),
    
    'conv2_1' : tf.constant(vgg16_weights[5]),
    'conv2_2' : tf.constant(vgg16_weights[7]),
    
    'conv3_1' : tf.constant(vgg16_weights[9]),
    'conv3_2' : tf.constant(vgg16_weights[11]),
    'conv3_3' : tf.constant(vgg16_weights[13]),
    
    'conv4_1' : tf.constant(vgg16_weights[15]),
    'conv4_2' : tf.constant(vgg16_weights[17]),
    'conv4_3' : tf.constant(vgg16_weights[19]),
    
    'conv5_1' : tf.constant(vgg16_weights[21]),
    'conv5_2' : tf.constant(vgg16_weights[23]),
    'conv5_3' : tf.constant(vgg16_weights[25]),
}

# input layer and output layer
x = tf.placeholder(tf.float32, [None, 160, 576, 3])
y = tf.placeholder(tf.float32, [None, 160, 576, 2])

def fcn(x, weights, biases):
    # First convolution layers
    conv1_1 = tf.nn.conv2d(x, 
                         weights['conv1_1'], 
                         strides = [1, 1, 1, 1], 
                         padding = 'SAME')
    conv1_1 = tf.nn.relu(tf.add(conv1_1, biases['conv1_1']))
    conv1_2 = tf.nn.conv2d(conv1_1, 
                         weights['conv1_2'], 
                         strides = [1, 1, 1, 1], 
                         padding = 'SAME')
    conv1_2 = tf.nn.relu(tf.add(conv1_2, biases['conv1_2']))
    maxp1 = tf.nn.max_pool(conv1_2, 
                           ksize = [1, 2, 2, 1], 
                           strides = [1, 2, 2, 1], 
                           padding = 'VALID')

    # Second convolution layers
    conv2_1 = tf.nn.conv2d(maxp1, 
                         weights['conv2_1'], 
                         strides = [1, 1, 1, 1], 
                         padding = 'SAME')
    conv2_1 = tf.nn.relu(tf.add(conv2_1, biases['conv2_1']))
    conv2_2 = tf.nn.conv2d(conv2_1, 
                         weights['conv2_2'], 
                         strides = [1, 1, 1, 1], 
                         padding = 'SAME')
    conv2_2= tf.nn.relu(tf.add(conv2_2, biases['conv2_2']))
    maxp2 = tf.nn.max_pool(conv2_2, 
                           ksize = [1, 2, 2, 1], 
                           strides = [1, 2, 2, 1], 
                           padding = 'VALID')

    # third convolution layers
    conv3_1 = tf.nn.conv2d(maxp2, 
                         weights['conv3_1'], 
                         strides = [1, 1, 1, 1], 
                         padding = 'SAME')
    conv3_1 = tf.nn.relu(tf.add(conv3_1, biases['conv3_1']))
    conv3_2 = tf.nn.conv2d(conv3_1, 
                         weights['conv3_2'], 
                         strides = [1, 1, 1, 1], 
                         padding = 'SAME')
    conv3_2= tf.nn.relu(tf.add(conv3_2, biases['conv3_2']))
    conv3_3 = tf.nn.conv2d(conv3_2, 
                         weights['conv3_3'], 
                         strides = [1, 1, 1, 1], 
                         padding = 'SAME')
    conv3_3= tf.nn.relu(tf.add(conv3_3, biases['conv3_3']))
    maxp3 = tf.nn.max_pool(conv3_3, 
                           ksize = [1, 2, 2, 1], 
                           strides = [1, 2, 2, 1], 
                           padding = 'VALID')

    # fourth convolution layers
    conv4_1 = tf.nn.conv2d(maxp3, 
                         weights['conv4_1'], 
                         strides = [1, 1, 1, 1], 
                         padding = 'SAME')
    conv4_1 = tf.nn.relu(tf.add(conv4_1, biases['conv4_1']))
    conv4_2 = tf.nn.conv2d(conv4_1, 
                         weights['conv4_2'], 
                         strides = [1, 1, 1, 1], 
                         padding = 'SAME')
    conv4_2= tf.nn.relu(tf.add(conv4_2, biases['conv4_2']))
    conv4_3 = tf.nn.conv2d(conv4_2, 
                         weights['conv4_3'], 
                         strides = [1, 1, 1, 1], 
                         padding = 'SAME')
    conv4_3= tf.nn.relu(tf.add(conv4_3, biases['conv4_3']))
    maxp4 = tf.nn.max_pool(conv4_3, 
                           ksize = [1, 2, 2, 1], 
                           strides = [1, 2, 2, 1], 
                           padding = 'VALID')

    # fifth convolution layers
    conv5_1 = tf.nn.conv2d(maxp4, 
                         weights['conv5_1'], 
                         strides = [1, 1, 1, 1], 
                         padding = 'SAME')
    conv5_1 = tf.nn.relu(tf.add(conv5_1, biases['conv5_1']))
    conv5_2 = tf.nn.conv2d(conv5_1, 
                         weights['conv5_2'], 
                         strides = [1, 1, 1, 1], 
                         padding = 'SAME')
    conv5_2= tf.nn.relu(tf.add(conv5_2, biases['conv5_2']))
    conv5_3 = tf.nn.conv2d(conv5_2, 
                         weights['conv5_3'], 
                         strides = [1, 1, 1, 1], 
                         padding = 'SAME')
    conv5_3= tf.nn.relu(tf.add(conv5_3, biases['conv5_3']))
    maxp5 = tf.nn.max_pool(conv5_3, 
                           ksize = [1, 2, 2, 1], 
                           strides = [1, 2, 2, 1], 
                           padding = 'VALID')
    
    # sixth convolution layer
    conv6 = tf.layers.conv2d(maxp5,
                            filters = 4096,
                            kernel_size = 7,
                            padding = 'SAME',
                            activation = tf.nn.relu)
    
    # 1x1 convolution layers
    fcn4 = tf.layers.conv2d(conv6,
                            filters = 4096,
                            kernel_size = 1,
                            padding = 'SAME',
                            activation = tf.nn.relu)
    fcn3 = tf.layers.conv2d(fcn4,
                            filters = 2,
                            kernel_size = 1,
                            padding = 'SAME')

    # Upsampling layers
    fcn2 = tf.layers.conv2d_transpose(fcn3,
                                      filters = 512,
                                      kernel_size = 4,
                                      strides = (2, 2),
                                      padding = 'SAME')
    
    fcn1 = tf.layers.conv2d_transpose(fcn2 + maxp4,
                                      filters = 256,
                                      kernel_size = 4,
                                      strides = (2, 2),
                                      padding = 'SAME')
    
    output = tf.layers.conv2d_transpose(fcn1 + maxp3, 
                                     filters = 2,
                                     kernel_size = 16,
                                     strides = (8, 8),
                                     padding = 'SAME')

    return output

LR  = 0.001

pred = fcn(x, weights, biases)
logits = tf.reshape(pred, (-1, 2))
labels = tf.reshape(y, (-1, 2))

loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
loss = tf.reduce_mean(loss)

optm  = tf.train.AdamOptimizer(LR).minimize(loss)

def train_batch_maker(batch_size):
    random_idx = np.random.randint(n_train, size = batch_size)
    return train_imgs[random_idx], train_seg[random_idx]
    
def test_batch_maker(batch_size):
    random_idx = np.random.randint(n_test, size = batch_size)
    return test_imgs[random_idx]
    
n_batch = 20
n_epoch = 300
n_prt = 30

sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss_record_train = []
for epoch in range(n_epoch):
    train_x, train_y = train_batch_maker(n_batch)
    sess.run(optm, feed_dict = {x: train_x, y: train_y})

    if epoch % n_prt == 0:        
        c = sess.run(loss, feed_dict = {x: train_x, y: train_y})
        loss_record_train.append(c)
        print ("Epoch : {}".format(epoch))
        print ("Cost : {}".format(c))
        
plt.figure(figsize = (10,8))
plt.plot(np.arange(len(loss_record_train))*n_prt, loss_record_train, label = 'training')
plt.xlabel('epoch', fontsize = 15)
plt.ylabel('loss', fontsize = 15)
plt.legend(fontsize = 12)
plt.ylim([0, np.max(loss_record_train)])
plt.show()

test_x = test_batch_maker(1)

test_img = sess.run(tf.nn.softmax(logits), feed_dict = {x: test_x})
test_img = test_img[:, 1].reshape(160, 576)
segmentation = (test_img > 0.5).reshape(160, 576, 1)

mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
mask = Image.fromarray((mask).astype('uint8'), mode = "RGBA")

plt.figure(figsize = (16,8))
plt.imshow(mask)
plt.axis('off')
plt.show() 

street_im = Image.fromarray((test_x[0] * 255).astype('uint8'))
street_im.paste(mask, box = None, mask = mask)

plt.figure(figsize = (16,8))
plt.imshow(street_im)
plt.axis('off')
plt.show() 


