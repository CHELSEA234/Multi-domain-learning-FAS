import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from warp import tf_batch_map_offsets

class Conv(layers.Layer):
    # GX: conv+bnorm+relu+dropout
    def __init__(self, ch=32, ksize=3, stride=1, norm='batch', nl=True, dropout=False, name=None):
        super(Conv, self).__init__()
        self.norm = norm
        self.conv = layers.Conv2D(ch, (ksize, ksize), strides=(stride, stride), padding='same',name=name)
        if norm == 'batch':
            # conv + bn
            self.bnorm = layers.BatchNormalization()
        else:
            self.bnorm = None
        if norm == 'spec':
            # conv + sn
            self.conv = tfa.layers.SpectralNormalization(self.conv)
        # relu
        if nl:
            self.relu = layers.LeakyReLU()
        else:
            self.relu = None
        # dropout
        if dropout:
            self.drop = layers.Dropout(0.3)
        else:
            self.drop = None

    def call(self, x, training):
        ## why you apply dropout during the training. 
        x = self.conv(x)
        if self.bnorm:
            x = self.bnorm(x, training)
        if self.relu:
            x = self.relu(x)
        if self.drop:
            x = self.drop(x)
        return x

class ConvT(layers.Layer):
    # GX: transpose+bnorm+relu+dropout
    def __init__(self, ch=32, ksize=3, stride=2, norm='batch', nl=True, dropout=False):
        super(ConvT, self).__init__()
        self.norm = norm
        self.conv = layers.Conv2DTranspose(ch, (ksize, ksize), strides=(stride, stride), padding='same')
        if norm == 'batch': # conv + bn
            self.bnorm = layers.BatchNormalization()
        else:
            self.bnorm = None
        if norm == 'spec': # conv + sn
            self.conv = tfa.layers.SpectralNormalization(self.conv)
        if nl: # relu
            self.relu = layers.LeakyReLU()
        else:
            self.relu = None
        if dropout: # dropout
            self.drop = layers.Dropout(0.3)
        else:
            self.drop = None

    def call(self, x, training):
        x = self.conv(x)
        if self.bnorm:
            x = self.bnorm(x, training)
        if self.relu:
            x = self.relu(x)
        if self.drop:
            x = self.drop(x)
        return x

class ShareLayer(layers.Layer):
    # GX: not in use.
    def __init__(self):
        super(ShareLayer, self).__init__(self)
        self.imsize = 256

    def call(self, x, reg, chuck):
        reg_in, reg_out = tf.split(reg, 2, axis=3)
        x_reg = tf_batch_map_offsets(x, reg_in)
        # reshape
        chuck_bsize,w,h,ch = x_reg.shape
        x_reg = tf.reshape(x_reg,[chuck_bsize//chuck,chuck,w,h,ch])
        x_max = tf.reduce_max(x_reg, axis=1)
        x_mean = tf.reduce_mean(x_reg, axis=1)
        x_share = tf.concat([x_max, x_mean], axis=3)
        x_share = tf.stack([x_share for _ in range(chuck)], axis=1)
        x_share = tf.reshape(x_share, [chuck_bsize,w,h,-1])
        x_share_dereg = tf_batch_map_offsets(x_share, reg_out)
        return x_share_dereg

class SA(layers.Layer):
    # GX: conv with differ ksize on the concatenated input.
    def __init__(self, ksize=3):
        super(SA, self).__init__()
        self.conv1 = Conv(1, ksize=ksize, name='conv')

    def call(self, x, training):
        xmean = tf.reduce_mean(x, axis=3, keepdims=True)
        xmax  = tf.reduce_max(x, axis=3,  keepdims=True)
        xmeanmax = tf.concat([xmean, xmax], axis=3)
        y = self.conv1(xmeanmax, training)
        return x*tf.sigmoid(y)

class region_estimator(tf.keras.Model):
    def __init__(self):
        super(region_estimator, self).__init__()
        self.up1 = ConvT(64)
        self.up2 = ConvT(40)
        self.up3 = ConvT(40)
        self.up4 = ConvT(40)
        self.conv_map1 = Conv(1, ksize=7, norm=False, nl=False)

    def call(self, feature, training):
        x1 = self.up1(feature[3])
        x2 = self.up2(tf.concat([x1, feature[2]], axis=3), training)
        x3 = self.up3(tf.concat([x2, feature[1]], axis=3), training)
        x4 = self.up4(tf.concat([x3, feature[0]], axis=3), training)
        output_f = self.conv_map1(x4)
        region_map = tf.nn.sigmoid(output_f)
        return region_map, [x1,x2,x3,x4]

class Generator(tf.keras.Model):
    def __init__(self, Region_E=None):
        super(Generator, self).__init__()
        self.RE = Region_E
        n_ch = [32,40,64,96,128,192,256]
        self.n_ch = n_ch
        self.conv0 = Conv(n_ch[0], ksize=7, name='conv0')
        self.conv1 = Conv(n_ch[1], name='conv1')
        self.conv2 = Conv(n_ch[1], name='conv2')
        self.conv3 = Conv(n_ch[1], name='conv3')
        self.conv4 = Conv(n_ch[1], name='conv4')
        self.conv5 = Conv(n_ch[1], name='conv5')
        self.conv6 = Conv(n_ch[1], name='conv6')
        self.conv7 = Conv(n_ch[1], name='conv7')
        self.conv8 = Conv(n_ch[1], name='conv8')
        self.conv9  = Conv(1, ksize=7, norm=False, nl=False, name='conv9')
        self.conv10 = Conv(3, ksize=7, norm=False, nl=False, name='conv10')
        self.conv11 = Conv(n_ch[4], name='conv11')
        self.conv12 = Conv(2, ksize=7, norm=False, nl=False, name='conv12')
        self.conv13 = Conv(3, ksize=7, norm=False, nl=False, name='conv13')

        self.up1 = ConvT(n_ch[2])
        self.up2 = ConvT(n_ch[1])
        self.up3 = ConvT(n_ch[1])
        self.up4 = ConvT(n_ch[1])
        self.up5 = ConvT(n_ch[2])
        self.up6 = ConvT(n_ch[1])
        self.up7 = ConvT(n_ch[1])
        self.up8 = ConvT(n_ch[1])
        self.up9 = ConvT(n_ch[2])
        self.up10 = ConvT(n_ch[1])
        self.up11 = ConvT(n_ch[1])
        self.up12 = ConvT(n_ch[1])

        self.down1 = Conv(n_ch[4], stride=2)
        self.down2 = Conv(n_ch[4], stride=2)
        self.down3 = Conv(n_ch[4], stride=2)
        self.down4 = Conv(n_ch[4], stride=2)

        self.sa1 = SA(7)
        self.sa2 = SA(5)
        self.sa3 = SA(3)
        self.sa4 = SA(3)
        self.sa5 = SA(7)
        self.sa6 = SA(7)

        self.pool2 = layers.AveragePooling2D(pool_size=(2, 2), strides=2)
        self.pool4 = layers.AveragePooling2D(pool_size=(4, 4), strides=4)
        self.pool8 = layers.AveragePooling2D(pool_size=(8, 8), strides=8)

    def call(self, img, training):
        ## GX: image with different scales.
        im_128 = tf.image.resize(self.pool2(img), [256, 256])
        im_64  = tf.image.resize(self.pool4(img), [256, 256])
        im_32  = tf.image.resize(self.pool8(img), [256, 256])

        ## GX: create some residual images.
        imgd1 = img - im_128
        imgd2 = im_128 - im_64
        imgd3 = im_64 - im_32
        imgd4 = im_32

        inputs = tf.concat([imgd1*25, imgd2*15, imgd3*8, imgd4], axis=3) 
        # (14, 256, 256, 12)
        # print(inputs.get_shape())
        # import sys;sys.exit(0)
        # nlayers = [16, 32, 40, 80, 120, 240, 480]
        x0 = self.conv0(inputs, training)
        # print(x0.get_shape())   # (14, 256, 256, 32)

        # x1_1 = func(x0); x1_2 = func'(x0, x1_1); x1_3 = func''(x0,x1_1,x1_2)
        # x1_3 = func'''(x1_2)
        x1_1 = tf.concat([x0, self.conv1(x0, training)], axis=3)
        # print(x1_1.get_shape()) # (14, 256, 256, 72) [40+32]
        x1_2 = tf.concat([x0, x1_1, self.conv2(x1_1, training)],axis=3)
        # print(x1_2.get_shape()) # (14, 256, 256, 144) 
        x1_3 = self.down1(x1_2, training)
        # print(x1_3.get_shape()) # (14, 128, 128, 128)

        x2_1 = tf.concat([x1_3, self.conv3(x1_3, training)], axis=3)
        # print(x2_1.get_shape()) # (14, 128, 128, 168)
        x2_2 = tf.concat([x1_3, x2_1, self.conv4(x2_1, training)],axis=3)
        # print(x2_2.get_shape()) # (14, 128, 128, 336)
        x2_3 = self.down2(x2_2, training)
        # print(x2_3.get_shape()) # (14, 64, 64, 128)

        x3_1 = tf.concat([x2_3, self.conv5(x2_3, training)], axis=3)
        # print(x3_1.get_shape()) # (14, 64, 64, 168)
        x3_2 = tf.concat([x2_3, x3_1, self.conv6(x3_1, training)],axis=3)
        # print(x3_2.get_shape()) # (14, 64, 64, 336)
        x3_3 = self.down3(x3_2, training)
        # print(x3_3.get_shape()) # (14, 32, 32, 128)

        x4_1 = tf.concat([x3_3, self.conv7(x3_3, training)], axis=3)
        # print(x4_1.get_shape()) # (14, 32, 32, 168)
        x4_2 = tf.concat([x3_3, x4_1, self.conv8(x4_1, training)],axis=3)
        # print(x4_2.get_shape()) # (14, 32, 32, 336)
        x4_3 = self.down4(x4_2, training)
        # print(x4_3.get_shape()) # (14, 16, 16, 128)
        region_map, output_feature = self.RE([x1_3,x2_3,x3_3,x4_3], training=training)
        ## don't need multiplication here.
        # x1_3 = 1e-4*x1_3*tf.image.resize(region_map, [128, 128]) + x1_3
        # x2_3 = 1e-4*x2_3*tf.image.resize(region_map, [64, 64]) + x2_3
        # x3_3 = 1e-4*x3_3*tf.image.resize(region_map, [32, 32]) + x3_3
        # x4_3 = 1e-4*x4_3*tf.image.resize(region_map, [16, 16]) + x4_3

        # u, w, v are for the p, n, c respectively.
        u1 = self.up1(x4_3, training)
        u2 = self.up2(tf.concat([u1, x3_3], axis=3), training)
        u3 = self.up3(tf.concat([u2, x2_3], axis=3), training)
        u4 = self.up4(tf.concat([u3, x1_3], axis=3), training)

        w1 = self.up5(x4_3, training)
        w2 = self.up6(tf.concat([w1, x3_3], axis=3), training)
        w3 = self.up7(tf.concat([w2, x2_3], axis=3), training)
        w4 = self.up8(tf.concat([w3, x1_3], axis=3), training)

        v1 = self.up9(x4_3, training)
        v2 = self.up10(v1, training)
        v3 = self.up11(v2, training)
        v4 = self.up12(v3, training)

        # print(u4.get_shape())   # (14, 256, 256, 40)
        # print(w4.get_shape())   # (14, 256, 256, 40)
        # print(v4.get_shape())   # (14, 256, 256, 40)
        # import sys;sys.exit(0)

        ## GX: one way is to merge p,n,c into one single trace.
        ## GX: conv9 and conv13 are different w.r.t kernel size.
        p  = tf.nn.sigmoid(self.conv9(u4, training))    # region
        n  = tf.nn.tanh(self.conv10(w4, training)/3e2)  # additive trace.
        c  = tf.nn.sigmoid(self.conv13(v4, training))   # content

        # ShortCut
        d1 = tf.image.resize(self.sa1(x1_3),[32,32])
        d2 = tf.image.resize(self.sa2(x2_3),[32,32])
        d3 = tf.image.resize(self.sa3(x3_3),[32,32])
        d4 = tf.image.resize(self.sa4(x4_3),[32,32])

        ## GX: what if I concat u4 and w4.
        ## GX: can you think how to modify the architecture in this?
        d5 = tf.image.resize(self.sa5(tf.stop_gradient(u4)),[32,32])
        d6 = tf.image.resize(self.sa6(tf.stop_gradient(w4)),[32,32])
        # GX: option1; you can go for different trace; 
        # GX: or you can even synthesize the continous map.
        ds = tf.concat([d1, d2, d3, d4, d5, d6],3)
        x4   = self.conv11(ds, training)
        dmap = self.conv12(x4, training)

        return dmap, p, c, n, output_feature, region_map

class Discriminator(tf.keras.Model):
    def __init__(self, downsize=1, num_layers=3):
        super(Discriminator, self).__init__()
        n_ch = [32,64,96,128,128,256]
        self.conv1 = Conv(n_ch[0], ksize=4, stride=2, norm=False)
        self.conv_stack = []
        for i in range(num_layers):
            self.conv_stack.append(Conv(n_ch[i], ksize=4, stride=2, norm='batch'))
        self.conv2 = Conv(n_ch[2], ksize=4, norm=False, nl=False)
        self.downsize = downsize
        self.num_layers = num_layers
    def call(self, x, training):
        if self.downsize > 1:
            _,w,h,_ = x.shape 
            x=tf.image.resize(x,(w//self.downsize,h//self.downsize))
        # print(x.get_shape())
        x = self.conv1(x, training)
        # print(x.get_shape())
        for i in range(self.num_layers):
            x = self.conv_stack[i](x, training)
        x = self.conv2(x)
        return tf.split(x,2,axis=0)

class Multi_Discriminator(tf.keras.Model):
    def __init__(self):
        super(Multi_Discriminator, self).__init__()
        n_ch = [32,64,96,128,128,256]
        self.conv1 = Conv(n_ch[0], ksize=4, stride=2, norm=False)
        self.conv_batch = Conv(n_ch[0], ksize=4, stride=2, norm='batch')
        self.conv2 = Conv(n_ch[2], ksize=4, norm=False, nl=False)
        self.conv3 = Conv(1, ksize=4, norm=False, nl=False)
        self.flat = tf.keras.layers.Flatten()
        self.fc   = tf.keras.layers.Dense(1)
    def call(self, x, training, last_output=False):
        _,w,h,_ = x.shape 
        # x = tf.image.resize(x,(16,16))
        x=tf.image.resize(x,(32,32))
        # print("============================")
        # print(x.get_shape())
        # print(self.conv1)
        x = self.conv1(x, training)
        # print("============================")
        # x = self.conv_batch(x, training)
        # x = self.conv2(x)
        x = self.conv3(x)
        if last_output:
            x = self.flat(x)
            x = tf.sigmoid(self.fc(x))
        return tf.split(x,2,axis=0)

class Feature_transform_layer(tf.keras.Model):
    def __init__(self, input_dim):
        super(Feature_transform_layer, self).__init__()
        self.conv1 = Conv(input_dim, ksize=4, stride=2, norm=False)
        self.conv2 = Conv(input_dim, ksize=4, stride=2, norm=False)

    def call(self, x):
        return self.conv2(self.conv1(x))

class Multi_Con_Discriminator(tf.keras.Model):
    def __init__(self):
        super(Multi_Con_Discriminator, self).__init__()
        n_ch = [32,64,96,128,128,256]
        self.conv1 = Conv(n_ch[0], ksize=4, stride=2, norm=False)
        self.conv_batch = Conv(n_ch[0], ksize=4, stride=2, norm='batch')
        self.conv2 = Conv(n_ch[2], ksize=4, norm=False, nl=False)
        self.conv3 = Conv(24, ksize=4, norm=False, nl=False)
        self.conv4 = Conv(1, ksize=4, norm=False, nl=False)
        self.flat = tf.keras.layers.Flatten()
        self.fc   = tf.keras.layers.Dense(1)
    def call(self, x, y, training, first_input=False, last_output=False):
        _,w,h,_ = x.shape 
        # x = tf.image.resize(x,(16,16))
        x=tf.image.resize(x,(32,32))
        if not first_input:
            y = tf.image.resize(y,(32,32))
            x = tf.concat([x,y], axis=3)
            # print(x.get_shape())
            # print("...over...")
            # import sys;sys.exit(0)
        # print("============================")
        # print(x.get_shape())
        # print(self.conv1)
        x = self.conv1(x, training)
        # print("============================")
        x = self.conv_batch(x, training)
        x = self.conv3(x)
        if last_output:
            x = self.conv4(x)
            x = self.flat(x)
            x = tf.sigmoid(self.fc(x))
            return tf.split(x,2,axis=0)
        return x
