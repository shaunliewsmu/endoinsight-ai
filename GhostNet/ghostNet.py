from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape, Dropout, Dense, Flatten
# from keras.utils.vis_utils import plot_model
# from tensorflow.keras import optimizers, regularizers
# from tensorflow.keras.layers import BatchNormalization

# from ghost_module import GhostModule
from GhostNet.ghost_module import GhostModule

class GhostNet(GhostModule):
    def __init__(self, shape, n_class, include_top=True):
        """Init"""
        super(GhostNet, self).__init__(shape, n_class)
        self.ratio = 2
        self.dw_kernel = 3
        self.include_top = include_top

    def build(self, plot=False):
        """GhostNet"""
        inputs = Input(shape=self.shape)

        x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2))
        print("Conv: shape = ", x.shape)

        x = self._ghost_bottleneck(x, 16, (3, 3), self.dw_kernel, 16, 1, self.ratio, False, name='ghost_bottleneck1')
        x = self._ghost_bottleneck(x, 24, (3, 3), self.dw_kernel, 48, 2, self.ratio, False, name='ghost_bottleneck2')

        x = self._ghost_bottleneck(x, 24, (3, 3), self.dw_kernel, 72, 1, self.ratio, False, name='ghost_bottleneck3')
        x = self._ghost_bottleneck(x, 40, (5, 5), self.dw_kernel, 72, 2, self.ratio, True, name='ghost_bottleneck4')

        x = self._ghost_bottleneck(x, 40, (5, 5), self.dw_kernel, 120, 1, self.ratio, True, name='ghost_bottleneck5')
        x = self._ghost_bottleneck(x, 80, (3, 3), self.dw_kernel, 240, 2, self.ratio, False, name='ghost_bottleneck6')

        x = self._ghost_bottleneck(x, 80, (3, 3), self.dw_kernel, 200, 1, self.ratio, False, name='ghost_bottleneck7')
        x = self._ghost_bottleneck(x, 80, (3, 3), self.dw_kernel, 184, 1, self.ratio, False, name='ghost_bottleneck8')
        x = self._ghost_bottleneck(x, 80, (3, 3), self.dw_kernel, 184, 1, self.ratio, False, name='ghost_bottleneck9')
        x = self._ghost_bottleneck(x, 112, (3, 3), self.dw_kernel, 480, 1, self.ratio, True, name='ghost_bottleneck10')
        x = self._ghost_bottleneck(x, 112, (5, 5), self.dw_kernel, 672, 1, self.ratio, True, name='ghost_bottleneck11')
        x = self._ghost_bottleneck(x, 160, (5, 5), self.dw_kernel, 672, 2, self.ratio, True, name='ghost_bottleneck12')

        x = self._ghost_bottleneck(x, 160, (5, 5), self.dw_kernel, 960, 1, self.ratio, False, name='ghost_bottleneck13')
        x = self._ghost_bottleneck(x, 160, (5, 5), self.dw_kernel, 960, 1, self.ratio, True, name='ghost_bottleneck14')
        x = self._ghost_bottleneck(x, 160, (5, 5), self.dw_kernel, 960, 1, self.ratio, False, name='ghost_bottleneck15')
        x = self._ghost_bottleneck(x, 160, (5, 5), self.dw_kernel, 960, 1, self.ratio, True, name='ghost_bottleneck16')

        x = self._conv_block(x, 960, (1, 1), strides=1)

        x = GlobalAveragePooling2D()(x)
        x = Reshape((1,1,960))(x)

        x = self._conv_block(x, 1280, (1,1), strides=1)

        x = Dropout(rate=0.05)(x)

        # my code
        x = Flatten()(x)

        if self.n_class == 1:
          x = Dense(self.n_class, activation='sigmoid')(x) # class = 1
        else:
          x = Dense(self.n_class, activation='softmax')(x) # class = 2

        # x = Conv2D(self.n_class, (1, 1), strides=1, padding='same',
        #            data_format='channels_last',name='last_Conv',
        #            activation='softmax', use_bias=False)(x)
        # print(x.shape)
       
        # if self.include_top:
        #     x = Reshape((self.n_class,))(x)

        model = Model(inputs, x)

        # if plot:
        #     plot_model(model, to_file='./images/GhostNet.png',show_shapes=True)

        return model