'''
Python code Implementation: Kernel Based Pooling Layer
'''

__author__ = "tchaton"


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement=True
session = tf.Session(config=config)
from keras.layers.pooling import _Pooling2D, InputSpec
import keras
from keras.legacy import interfaces
from keras import initializers
from keras.layers import regularizers, constraints, activations
from keras import backend as K

class KernelBasedPooling(_Pooling2D):

    '''
	Perform N conv2d with size kernel_size over each channel size, average them and concat them
    '''


    @interfaces.legacy_conv2d_support
    def __init__(self, filters, kernel_size=(2, 2), strides=None, padding='valid', data_format=None,              
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 activation='linear',
		 with_softmax=True,
                 **kwargs):
        super(KernelBasedPooling, self).__init__(kernel_size, strides, padding,
                                           data_format, **kwargs)
        self.rank = 2
	self.with_softmax = with_softmax
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
	print(self.kernel.get_shape())
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
        

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):

        channels = inputs.get_shape()[-1]
        holder = []
        for c in range(channels):
            init = tf.expand_dims(inputs[:,:,:,c], axis=-1)
            if self.rank == 2:
                outputs = K.conv2d(
                    init,
                    tf.expand_dims(self.kernel[:,:,c,:], 2),
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format)
		if self.with_softmax:
		    exp = tf.exp(outputs)	
		    prob = exp / tf.reduce_sum(exp, -1, keep_dims=True)
                    outputs = outputs * prob
                holder.append(tf.expand_dims(K.mean(outputs, axis=-1), axis=-1))
        return tf.concat(holder, axis=-1) 

if __name__ == '__main__':

    kbp = KernelBasedPooling(4)
    kbp.build([1, 224, 224, 3])
    init_var = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run([init_var])
        init = tf.random_normal([1, 224, 224, 3],
                                mean=0.0,
                                stddev=1.0,
                                dtype=tf.float32,
                                seed=42,
				)
    	output = kbp(init)
    	out = sess.run(output)

	print(out.shape)
	'''
		Result : 1/ (1, 224, 224, 3) : [(1, 224, 224, 1) for _ in range(3)] #Split over channel dimension 
                              ->(conv2d)->                                          # Apply the conv2 kernel over each element in the list
                         2/ [(1, 112, 112, 4) for _ in range(3)]                   # Output shape of the convolution
                              ->(mean)->                                            # Take average over channel dimension
                         3/ (1, 112, 112, 3)                                        # Final output size (same as AveragePooling2D, MaxPooling) but we are learning lot of diffents way to pool
	'''
