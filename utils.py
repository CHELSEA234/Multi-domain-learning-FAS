from skimage.draw import line_aa
## place this before from skimage.draw import line_aa, will generate error...
import cv2
import tensorflow as tf
import sys
import glob
import random
import numpy as np
import math as m
import tensorflow.keras.layers as layers

import matplotlib.tri as mtri
from scipy import ndimage, misc
from PIL import Image, ImageDraw

class Logging(object):
  def __init__(self, config):
    self.config = config
    self.losses = {}
    self.losses_val = {}
    self.txt = ''
    self.fig = []
    self.fig_val = []

  def update(self, losses, training):
    if training:
      for name in losses.keys():
          if name in self.losses:
              current_loss = self.losses[name]
              self.losses[name] = [current_loss[0]+losses[name], current_loss[1]+1]
          else:
              self.losses[name] = [losses[name], 1]
    else:
      for name in losses.keys():
          if name in self.losses_val:
              current_loss = self.losses_val[name]
              self.losses_val[name] = [current_loss[0]+losses[name].numpy(), current_loss[1]+1]
          else:
              self.losses_val[name] = [losses[name].numpy(), 1]

  def display(self, losses, epoch, step, training, allstep):
    self.update(losses, training)
    if training:
        text = 'Epoch (Train) '+str(epoch+1)+'-'+str(step+1)+'/'+str(allstep) + ': '
        for _name in self.losses.keys():
            value = self.losses[_name]
            text += _name+':'+"{:.3g}".format(value[0]/value[1])+', '
    else:
        text = 'Epoch ( Val ) '+str(epoch+1)+'-'+str(step+1)+'/'+str(allstep) + ': '
        for _name in self.losses_val.keys():
            value = self.losses_val[_name]
            text += _name+':'+"{:.3g}".format(value[0]/value[1])+', '
    
    text = text[:-2]+'     '
    # display loss
    print(text, end='\r')
    self.txt = text
    self.epoch = epoch
    self.step = step

  ## you should display the 
  def display_metric(self, message):
    config = self.config
    print(message, end='\r')
    file_object = open(config.CHECKPOINT_DIR+'/log.txt', 'a')
    file_object.write(message+'\n')
    file_object.close()

  def save(self, fig, training):
    config = self.config
    step = self.step

    ## GX: this one is important...
    ## GX: this explains why your program is slow here...
    # fig = self.get_figures(fig)

    if training:
        if step % config.IMG_LOG_FR == 0:
          fig = self.get_figures(fig)
          fname = config.CHECKPOINT_DIR + '/epoch-' + str(self.epoch+1) + '-Train-' + str(self.step+1) + '.png'
          cv2.imwrite(fname, fig.numpy())
        if step % config.TXT_LOG_FR == 0:
          file_object = open(config.CHECKPOINT_DIR+'/log.txt', 'a')
          file_object.write(self.txt+'\n')
          file_object.close()
    else:
        if step % (config.IMG_LOG_FR//10) == 0:
          fig = self.get_figures(fig)
          fname = config.CHECKPOINT_DIR + '/epoch-' + str(self.epoch+1) + '-Val-' + str(self.step+1) + '.png'
          cv2.imwrite(fname, fig.numpy())
        if step % (config.TXT_LOG_FR//10) == 0:
          file_object = open(config.CHECKPOINT_DIR+'/log.txt', 'a')
          file_object.write(self.txt+'\n')
          file_object.close()
    self.fig = []
    self.fig_val = []

  def save_img(self, fig, fname):
    config = self.config
    step = self.step
    fig = self.get_imgs(fig,256)
    fname = config.CHECKPOINT_DIR+'/test/'+fname.split('/')[-1].split('.')[0]+'-result.png'
    cv2.imwrite(fname, fig.numpy())
    self.fig = []
    self.fig_val = []

  def reset(self):
    losses = {}
    losses_val = {}
    ind = 0
    for _name in self.loss_names:
      self.losses[_name] = [0, 0]
      self.losses_val[_name] = [0, 0]
      ind += 1
    self.txt = ''
    self.img = 0

  def get_imgs(self, fig, size=None):
    config = self.config
    column = []
    for _img in fig:
      _img = tf.clip_by_value(_img, 0.0, 1.0)*255
      if _img.shape[3] == 1:
          _img = tf.concat([_img, _img, _img], axis=3)
      else:
          r, g, b = tf.split(_img[:,:,:,:3], 3, 3)
          _img = tf.concat([b,g,r], 3)
      if size is None:
        _img = tf.image.resize(_img, [config.FIG_SIZE, config.FIG_SIZE])
      else:
        _img = tf.image.resize(_img, [config.IMG_SIZE, config.IMG_SIZE])
      column.append(_img[0,:,:,:])
    column = tf.concat(column, axis=0)
    return column

  def get_figures(self, fig, size=None):
    config = self.config
    column = []
    for _img in fig:
      _img = tf.clip_by_value(_img, 0.0, 1.0)*255
      if _img.shape[3] == 1:
          _img = tf.concat([_img, _img, _img], axis=3)
      else:
          r, g, b = tf.split(_img[:,:,:,:3], 3, 3)
          _img = tf.concat([b,g,r], 3)
      if size is None:
        _img = tf.image.resize(_img, [config.FIG_SIZE, config.FIG_SIZE])
      else:
        _img = tf.image.resize(_img, [config.IMG_SIZE, config.IMG_SIZE])
      _row = tf.split(_img, _img.shape[0])
      _row = tf.concat(_row, axis=2)
      column.append(_row[0,:,:,:])
    column = tf.concat(column, axis=0)
    return column

def l1_loss(x, y, mask=None):
    xshape = x.shape
    if mask is not None:
        #loss = tf.math.reduce_mean(tf.reshape(tf.abs(tf.subtract(x, y)), [xshape[0], -1]), axis=1)
        loss = tf.math.reduce_sum(tf.abs(x-y) * mask) / (tf.reduce_sum(mask) + 1e-6) / x.shape[3]
    else:
        loss = tf.math.reduce_mean(tf.abs(x-y))
    return loss

def l2_loss(x, y, mask=None):
    xshape = x.shape
    if mask is not None:
        loss = tf.math.reduce_sum(tf.square(tf.subtract(x, y)) * mask) / (tf.reduce_sum(mask) + 1e-6) / x.shape[3]
    else:
        loss = tf.math.reduce_mean(tf.square(tf.subtract(x, y)))
    return loss

def hinge_loss(y_pred, y_true, mask=None):
    return tf.math.reduce_mean(tf.math.maximum(0., 1. - y_true*y_pred))

def generate_face_region(source, img_size):
    morelm = np.copy(source[0:17,:])
    morelm[:,1] = morelm[0,1] - (morelm[:,1] - morelm[0,1]) * 0.8
    source = np.concatenate([source,morelm],axis=0)
    '''
    img = Image.new('L', (img_size, img_size), 0)
    ImageDraw.Draw(img).polygon(source, outline=1, fill=1)
    mask = np.array(img)
    mask = cv2.GaussianBlur(mask,(5,5),0).reshape([img_size,img_size,1])

    '''
    xi, yi = np.meshgrid(np.linspace(0, 1, img_size), np.linspace(0, 1, img_size))
    # interp2d
    _triang = mtri.Triangulation(source[:,0], source[:,1])
    _interpx = mtri.LinearTriInterpolator(_triang, source[:,0])
    _offsetmapx = _interpx(xi, yi)

    offsetmap = np.stack([_offsetmapx], axis=2)
    offsetmap = np.nan_to_num(offsetmap)  
    offsetmap = np.asarray(offsetmap>0,np.float32)
    offsetmap = cv2.GaussianBlur(offsetmap,(5,5),0).reshape([img_size,img_size,1])
    return offsetmap

def generate_landmark_map(landmark, img_size):
    lmlist = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],[16,17],
              [18,19],[19,20],[20,21],[21,22],[23,24],[24,25],[25,26],[26,27],
              [37,38],[38,39],[39,40],[40,41],[41,42],[42,37],[43,44],[44,45],[45,46],[46,47],[47,48],[48,43],
              [28,29],[29,30],[30,31],[32,33],[33,34],[34,35],[35,36],
              [49,50],[50,51],[51,52],[52,53],[53,54],[54,55],[55,56],[56,57],[57,58],[58,59],[59,60],[60,49],
              [61,62],[62,63],[63,64],[64,65],[65,66],[66,67],[67,68],[68,61]]
    lm_map = []
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    lm = landmark*img_size
    for pr in lmlist:
        lm_start = lm[pr[0]-1,:].astype(np.int32)
        lm_end = lm[pr[1]-1,:].astype(np.int32)
        rr, cc, val = line_aa(lm_start[0], lm_start[1], lm_end[0], lm_end[1])
        templist = [t for t in range(len(rr)) if rr[t] < img_size and rr[t] > 0 ]
        rr = rr[templist]
        cc = cc[templist]
        val = val[templist]
        templist = [t for t in range(len(cc)) if cc[t] < img_size and cc[t] > 0 ]
        rr = rr[templist]
        cc = cc[templist]
        val = val[templist]
        img[cc, rr] = val * 255
    blur = cv2.GaussianBlur(img,(3,3),0)
    blur = blur / np.amax(blur) * 255
    lm_map = np.reshape(blur, [blur.shape[0], blur.shape[1], 1])
    return lm_map

def face_crop_and_resize(img, lm, fsize, box_perturb=[1.15, 1.25], aug=False):
    ## visualize this function.
    img_shape = img.shape
    lm_reverse_list = np.array([17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,
                               27,26,25,24,23,22,21,20,19,18,
                               28,29,30,31,36,35,34,33,32,
                               46,45,44,43,48,47,40,39,38,37,42,41,
                               55,54,53,52,51,50,49,60,59,58,57,56,65,64,63,62,61,68,67,66],np.int32) -1                
    if aug and lm.shape[0] == 68 and random.uniform(0,1)>0.5:
        img = cv2.flip(img, 1)  # horizontal reverse.
        lm[:,0] = img_shape[1] - lm[:,0]
        lm = lm[lm_reverse_list,:]

    # center and length of the landmarks.
    center = [(np.min(lm[:,0])+np.max(lm[:,0]))/2, (np.min(lm[:,1])+np.max(lm[:,1]))/2]
    length = np.max([(np.max(lm[:,0])-np.min(lm[:,0]))/2, (np.max(lm[:,1])-np.min(lm[:,1]))/2]) * 1.1
    if aug: # if aug, change the center and length.
      center[0] = center[0] + random.uniform(-0.05,0.05)*length
      center[1] = center[1] + random.uniform(-0.05,0.05)*length
      length = length * random.uniform(0.93,1.07)
    ## cropping the image.
    box = [int(center[0])-int(length),
           int(center[1])-int(length*1.2),
           int(center[0])+int(length),
           int(center[1])+int(length)+int(length)-int(length*1.2)]
    # box0 = [int(center[0])-int(length),
    #        int(center[1])-int(length*1.2),
    #        int(center[0])+int(length),
    #        int(center[1])+int(length)+int(length)-int(length*1.2)]
    #print(box, img.shape)
    box_m = [img_shape[1] - box[2],
             box[1],
             img_shape[1] - box[0],
             box[3]]

    lm[:,0] = lm[:,0] - box[0]
    lm[:,1] = lm[:,1] - box[1]

    preset_x = 0
    preset_y = 0
    if box[0] < 0 or box[2] > img_shape[1]:
        preset_x = max(-box[0], box[2] - img_shape[1])
    if box[1] < 0 or box[3] > img_shape[0]:
        preset_y = max(-box[1], box[3] - img_shape[0])
    if preset_x > 0 or preset_y > 0:
        img_large= np.zeros((img_shape[0]+preset_y+preset_y+2,img_shape[1]+preset_x+preset_x+2,img_shape[2]))
        img_large[preset_y:preset_y+int(img_shape[0]),preset_x:preset_x+int(img_shape[1]),:] = img
        img = img_large
        box[0] = box[0] + preset_x
        box[1] = box[1] + preset_y
        box[2] = box[2] + preset_x
        box[3] = box[3] + preset_y
    img = img[box[1]:box[3],box[0]:box[2],:]
    sz = img.shape[0]
    if img.shape[0] == img.shape[1] and img.shape[0]>0:
        img = cv2.resize(img,   (fsize,fsize))
    else:
        img = np.zeros((fsize, fsize, img.shape[2])) 
    # print("...coming here...")
    # print(img.shape, type(img), np.max(img), np.min(img))
    # print("...coming here...")
    # import sys;sys.exit(0)
    return img, lm/(length*2)