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
import tensorflow as tf


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
        ngram_layers = self._ngram_conv_layers(32, 3, 'same', 'relu')
        print('3')
        left_ngrams = [layer(embed_left) for layer in ngram_layers]
        right_ngrams = [layer(embed_right) for layer in ngram_layers]
        print('3.5')
        left_idfs=[]
        for n in range(1,3):
            idf_tensor = tf.py_function(self.get_ngram_idf, [input_left, n], tf.dtypes.float32)
            idf_tensor.set_shape(input_left.get_shape())
            left_idfs.append(idf_tensor)
        rihgt_idfs = []
        for n in range(1, 3):
            idf_tensor = tf.py_function(self.get_ngram_idf, [input_right, n], tf.dtypes.float32)
            idf_tensor.set_shape(input_right.get_shape())
            rihgt_idfs.append(idf_tensor)

        print('4')
        right_idfs = [tf.py_function(self.get_ngram_idf, [input_right, n], tf.dtypes.float32) for n in range(1, 3)]
        print('6')
        matching_layer = matchzoo.layers.MatchingLayer(matching_type='dot')
        print('92')
        mask_layer = matchzoo.layers.MatchingLayer(matching_type='mul')
        print('94')
        ngram_product = [matching_layer([m, n]) for m in left_ngrams for n in right_ngrams]
        mask_tensors = [matching_layer([m, n]) for m in left_idfs for n in right_idfs]
        print('96')
        ngram_output = keras.layers.Concatenate(axis=-1, name='concate1')(ngram_product)
        mask_output = keras.layers.Concatenate(axis=-1, name='concate2')(mask_tensors)
        print('98')
        ngram_output = mask_layer([ngram_output, mask_output])
        print('100')
        for i in range(self._params['num_blocks']):
            ngram_output = self._conv_block(
                ngram_output,
                self._params['kernel_count'][i],
                self._params['kernel_size'][i],
                self._params['padding'],
                self._params['activation']
            )

        # Dynamic Pooling
        # dpool_layer = matchzoo.layers.DynamicPoolingLayer(
        #     *self._params['dpool_size'])
        pool_layer = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        embed_pool = pool_layer(ngram_output)
        embed_flat = keras.layers.Flatten()(embed_pool)
        x = keras.layers.Dropout(rate=self._params['dropout_rate'])(embed_flat)

        inputs = [input_left, input_right]
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
            activation: str
    ) -> typing.Any:
        layers = [keras.layers.Conv1D(kernel_count,
                                      kernel_size,
                                      padding=padding,
                                      activation=activation, name='ngram_conv1d_' + str(kernel_size)) for kernel_size in
                  range(1, n + 1)]
        return layers

    def input_to_term(self, _input: list) -> list:
        return [self._params['vocab_unit'].state['index_term'][i] for i in _input]

    def get_ngram_idf(self, _input, n: int) -> list:
        """
        padding
        """
        assert n > 0
        print('into getngram')
        padding_input = _input
        if n > 1:
            pad = np.array([[0]*(n-1)] * int(_input.shape[0]))
            padding_input = np.concatenate((_input,pad),axis=-1)
        uniidf = list(map(lambda x:self._params['vocab_unit'].state['index_term'][self._params['vocab_unit'].state['index_term'][int(x)]], padding_input))
        ngramidf = []
        for i in uniidf:
            ngramidf.append([max(i[x:x+n]) for x in range(len(i) - n + 1)])
        return np.array(ngramidf)
