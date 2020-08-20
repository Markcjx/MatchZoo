"""An implementation of MatchPyramid Model."""
import typing

import keras

import matchzoo
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine import hyper_spaces
from matchzoo.preprocessors.units import Vocabulary
import numpy as np
from keras import backend as K
from keras.layers import Lambda
from keras.layers import Input
import tensorflow as tf
import pdb


class Mix(BaseModel):
    """
    Mix Model.

    Examples:
        >>> model = MatchPyramid()
        >>> model.params['embedding_output_dim'] = 300
        >>> model.params['num_blocks'] = 2
        >>> model.params['kernel_count'] = [16, 32]
        >>> model.params['kernel_size'] = [[3, 3], [3, 3]]
        >>> model.params['dpool_size'] = [3, 10]
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params.add(Param(name='num_blocks', value=1,
                         desc="Number of convolution blocks."))
        params.add(Param(name='kernel_count', value=[32],
                         desc="The kernel count of the 2D convolution "
                              "of each block."))
        params.add(Param(name='kernel_size', value=[[3, 3]],
                         desc="The kernel size of the 2D convolution "
                              "of each block."))
        params.add(Param(name='activation', value='relu',
                         desc="The activation function."))
        params.add(Param(name='dpool_size', value=[3, 10],
                         desc="The max-pooling size of each block."))
        params.add(Param(
            name='padding', value='same',
            desc="The padding mode in the convolution layer."
        ))
        params.add(Param(
            name='dropout_rate', value=0.0,
            hyper_space=hyper_spaces.quniform(low=0.0, high=0.8,
                                              q=0.01),
            desc="The dropout rate."
        ))
        params.add(Param(
            name='vocab_unit', value=Vocabulary(), desc='the Vocabulary'
        ))
        params.add(Param(
            name='idf_table', value={}, desc='get terms idf'
        ))
        return params

    def build(self):
        """
        Build model structure.

        MatchPyramid text matching as image recognition.
        """
        print('1')
        input_left, input_right = self._make_inputs()
        pos_left = Input(  name='pos_left',
            shape=self._params['input_shapes'][0])
        pos_right = Input(name='pos_right',
                         shape=self._params['input_shapes'][1])

        print(pos_left.shape)
        print(pos_right.shape)
        # input_dpool_index = keras.layers.Input(
        #     name='dpool_index',
        #     shape=[self._params['input_shapes'][0][0],
        #            self._params['input_shapes'][1][0],
        #            2],
        #     dtype='int32')

        embedding = self._make_embedding_layer()
        embed_left = embedding(input_left)
        embed_right = embedding(input_right)
        # Interaction
        print('2')
        ngram_layer = self._ngram_conv_layers(32, 3, 'same', 'relu',name = 'common')
        left_ngrams = [layer(embed_left) for layer in ngram_layer]
        right_ngrams = [layer(embed_right) for layer in ngram_layer]
        matching_layer = matchzoo.layers.MatchingLayer(matching_type='dot')
        ngram_product = [matching_layer([m, n]) for m in left_ngrams for n in right_ngrams]
        print('3.5')
        left_idf = Lambda(self.convert_to_idf_tensor)(input_left)
        print(left_idf.shape)
        print('4')
        right_idf = Lambda(self.convert_to_idf_tensor)(input_right)
        print(left_idf.shape)
        print('8')
        # left_idf_arr = [keras.layers.MaxPooling1D(pool_size=n, strides=1, padding='same')(left_idf) for n in
        #                 range(1, 4)]
        # right_idf_arr = [keras.layers.MaxPooling1D(pool_size=n, strides=1, padding='same')(right_idf) for n in
        #                  range(1, 4)]
        # print('8')
        dot_layer = keras.layers.Dot(-1)
        multi_layer = keras.layers.Multiply()
        # idf_masks = [dot_layer([left, right]) for left in left_idf_arr for right in right_idf_arr]
        # reshape = keras.layers.Reshape(tuple(idf_masks[0].shape.as_list()[1:]) + (1,))
        # idf_masks = [reshape(idf_mask) for idf_mask in idf_masks]

        idf_mask = dot_layer([left_idf, right_idf])
        pos_mask = dot_layer([pos_left, pos_right])
        print(idf_mask.shape)
        print(pos_mask.shape)
        reshape = keras.layers.Reshape(tuple(idf_mask.shape.as_list()[1:]) + (1,))
        idf_mask = reshape(idf_mask)
        pos_mask = reshape(pos_mask)
        print('9')
        #
        # for i in [ngram_product]:
        #     for j in i:
        #         print(j.shape)
        ngram_product = [multi_layer([idf_mask, ngram_product[i]]) for i in range(len(ngram_product))]
        pos_product = [multi_layer([pos_mask, ngram_product[i]]) for i in range(len(ngram_product))]
        ngram_product.extend(pos_product)
        print('96')
        print('ngram_product shape is %s' % ngram_product[0].shape)
        ngram_output = keras.layers.Concatenate(axis=-1, name='concate1')(ngram_product)
        print(ngram_output.shape)
        print('100')
        for i in range(self._params['num_blocks']):
            ngram_output = self._conv_block(
                ngram_output,
                self._params['kernel_count'][i],
                self._params['kernel_size'][i],
                self._params['padding'],
                self._params['activation']
            )
        print('128')
        # Dynamic Pooling
        # dpool_layer = matchzoo.layers.DynamicPoolingLayer(
        #     *self._params['dpool_size'])
        pool_layer = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        print('129')
        embed_pool = pool_layer(ngram_output)
        print('130')
        embed_flat = keras.layers.Flatten()(embed_pool)
        print('131')
        x = keras.layers.Dropout(rate=self._params['dropout_rate'])(embed_flat)
        print('132')
        inputs = [input_left, input_right]
        print('133')
        x_out = self._make_output_layer()(x)
        self._backend = keras.Model(inputs=inputs, outputs=x_out)

    @classmethod
    def _conv_block(
            cls, x,
            kernel_count: int,
            kernel_size: int,
            padding: str,
            activation: str
    ) -> typing.Any:
        output = keras.layers.Conv2D(kernel_count,
                                     kernel_size,
                                     padding=padding,
                                     activation=activation)(x)
        return output

    @classmethod
    def _ngram_conv_layers(
            cls,
            kernel_count: int,
            n: int,
            padding: str,
            activation: str,
            name: str = '',
    ) -> typing.Any:
        layers = [keras.layers.Conv1D(kernel_count,
                                      kernel_size,
                                      padding=padding,
                                      activation=activation, name=name + '_ngram_conv_' + str(kernel_size)) for
                  kernel_size in
                  range(1, n + 1)]
        return layers

    def get_ngram_idf(self, _input):
        def trans_to_idf(x):
            term = self._params['vocab_unit'].state['index_term'].get(int(x))
            if not term:
                term = '<OOV>'
            try:
                idf = self._params['idf_table'][term]
            except:
                idf = 0.5
            return float(idf)

        trans_func = np.frompyfunc(trans_to_idf, 1, 1)
        uniidf = trans_func(_input)
        return np.array(uniidf)

    def get_pos_score(self,_input):
        def trans_to_pos_score(x):
            pos_score = self._params['pos'][x]
            return pos_score
        trans_func = np.frompyfunc(trans_to_pos_score, 1, 1)
        pos_array = trans_func(_input)
        return pos_array

    def mul(self, _input):
        assert len(_input) == 2
        left, right = _input
        print('left %s' % left.shape)
        print('right %s' % right.shape)
        return left * right

    def convert_to_idf_tensor(self, _input, ):
        idf_tensor = tf.py_function(self.get_ngram_idf, [_input], tf.dtypes.float32)
        idf_tensor.set_shape(_input.get_shape())
        print('idf_tensor shape1 %s ' % idf_tensor.shape)
        idf_tensor = tf.expand_dims(idf_tensor, 2)
        print('idf_tensor shape2 %s ' % idf_tensor.shape)
        return idf_tensor

    def convert_to_pos_tensor(self,_input):
        pos_tensor = tf.py_function(self.get_pos_score,[_input],tf.dtypes.float32)
        pos_tensor.set_shape(_input.get_shape())
        pos_tensor = tf.expand_dims(pos_tensor, 2)
        return pos_tensor