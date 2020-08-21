"""Basic Preprocessor."""

from tqdm import tqdm
from pathlib import Path
from . import units
from matchzoo import DataPack
from matchzoo.engine.base_preprocessor import BasePreprocessor
from .build_vocab_unit import build_mix_vocab_unit
from .build_unit_from_data_pack import build_unit_from_data_pack
from .chain_transform import chain_transform
import typing
import dill

tqdm.pandas()


class ChineseBasicPreprocessor(BasePreprocessor):
    """
    Baisc preprocessor helper.

    :param fixed_length_left: Integer, maximize length of :attr:`left` in the
        data_pack.
    :param fixed_length_right: Integer, maximize length of :attr:`right` in the
        data_pack.
    :param filter_mode: String, mode used by :class:`FrequenceFilterUnit`, Can
        be 'df', 'cf', and 'idf'.
    :param filter_low_freq: Float, lower bound value used by
        :class:`FrequenceFilterUnit`.
    :param filter_high_freq: Float, upper bound value used by
        :class:`FrequenceFilterUnit`.
    :param remove_stop_words: Bool, use :class:`StopRemovalUnit` unit or not.

    Example:
        >>> import matchzoo as mz
        >>> train_data = mz.datasets.toy.load_data('train')
        >>> test_data = mz.datasets.toy.load_data('test')
        >>> preprocessor = mz.preprocessors.BasicPreprocessor(
        ...     fixed_length_left=10,
        ...     fixed_length_right=20,
        ...     filter_mode='df',
        ...     filter_low_freq=2,
        ...     filter_high_freq=1000,
        ...     remove_stop_words=True
        ... )
        >>> preprocessor = preprocessor.fit(train_data, verbose=0)
        >>> preprocessor.context['input_shapes']
        [(10,), (20,)]
        >>> preprocessor.context['vocab_size']
        228
        >>> processed_train_data = preprocessor.transform(train_data,
        ...                                               verbose=0)
        >>> type(processed_train_data)
        <class 'matchzoo.data_pack.data_pack.DataPack'>
        >>> test_data_transformed = preprocessor.transform(test_data,
        ...                                                verbose=0)
        >>> type(test_data_transformed)
        <class 'matchzoo.data_pack.data_pack.DataPack'>

    """

    def __init__(self, fixed_length_left: int = 30,
                 fixed_length_right: int = 30,
                 filter_mode: str = 'df',
                 filter_low_freq: float = 2,
                 filter_high_freq: float = float('inf'),
                 remove_stop_words: bool = False):
        """Initialization."""
        super().__init__()
        self._fixed_length_left = fixed_length_left
        self._fixed_length_right = fixed_length_right
        self._left_fixedlength_unit = units.HanLP_Fix_length(
            self._fixed_length_left,
            pad_value='<PAD>',
            pad_mode='post',
            truncate_mode='post'
        )
        self._right_fixedlength_unit = units.HanLP_Fix_length(
            self._fixed_length_right,
            pad_value='<PAD>',
            pad_mode='post',
            truncate_mode='post'
        )
        self._filter_unit = units.FrequencyFilter(
            low=filter_low_freq,
            high=filter_high_freq,
            mode=filter_mode
        )
        self._units = self._custom_units()
        if remove_stop_words:
            self._units.append(units.ChineseStopRemoval())

    def fit(self, data_pack: DataPack, cunstom_idf, verbose: int = 1):
        """
        Fit pre-processing context for transformation.

        :param data_pack: data_pack to be preprocessed.
        :param verbose: Verbosity.
        :return: class:`BasicPreprocessor` instance.
        """
        data_pack = data_pack.apply_on_text(chain_transform(self._units),
                                            verbose=verbose)
        # fitted_filter_unit = build_unit_from_data_pack(self._filter_unit,
        #                                                data_pack,
        #                                                flatten=False,
        #                                                mode='right',
        #                                                verbose=verbose)
        # data_pack = data_pack.apply_on_text(fitted_filter_unit.transform,
        #                                     mode='right', verbose=verbose)
        # self._context['filter_unit'] = fitted_filter_unit

        vocab_unit = build_mix_vocab_unit(data_pack, cunstom_idf, verbose=verbose)
        self._context['vocab_unit'] = vocab_unit

        vocab_size = len(vocab_unit.state['term_index'])
        self._context['vocab_size'] = vocab_size
        self._context['embedding_input_dim'] = vocab_size
        self._context['input_shapes'] = [(self._fixed_length_left,),
                                         (self._fixed_length_right,)]

        return self

    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data, create fixed length representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        data_pack.apply_on_text(chain_transform(self._units), inplace=True,
                                verbose=verbose)

        # data_pack.apply_on_text(self._context['filter_unit'].transform,
        #                         mode='right', inplace=True, verbose=verbose)
        data_pack.append_text_length(inplace=True, verbose=verbose)
        data_pack.apply_on_text(self._left_fixedlength_unit.transform,
                                mode='left', inplace=True, verbose=verbose)
        data_pack.apply_on_text(self._right_fixedlength_unit.transform,
                                mode='right', inplace=True, verbose=verbose)
        data_pack.apply_on_text(self.get_part_of_speech, inplace=True, mode='both', rename=('pos_left', 'pos_right'))
        data_pack.apply_on_text(self._context['vocab_unit'].transform,
                                mode='both', inplace=True, verbose=verbose)

        max_len_left = self._fixed_length_left
        max_len_right = self._fixed_length_right

        data_pack.left['length_left'] = \
            data_pack.left['length_left'].apply(
                lambda val: min(val, max_len_left))

        data_pack.right['length_right'] = \
            data_pack.right['length_right'].apply(
                lambda val: min(val, max_len_right))
        return data_pack

    @classmethod
    def _custom_units(cls) -> list:
        """Prepare needed process units."""
        return [
            units.ChinesePuncRemoval(),
            units.Lowercase(False),
            units.HanLPTokenize()
        ]

    def get_part_of_speech(self, _input: list) -> list:
        """just for HanLP segment"""
        return [term if type(term) == str else term.nature.toString() for term in _input]

    def get_idf(self, _input: list):
        return [self._context['vocab_unit'].state['idf_table'][term] if type(term) == str else
                self._context['vocab_unit'].state['idf_table'][term.word] for term in _input]

    def save(self, dirpath: typing.Union[str, Path]):
        """
        Save the :class:`ChineseBasicPreprocessor` object.

        A saved :class:`ChineseBasicPreprocessor` is represented as a directory with
        the `context` object (fitted parameters on training data), it will
        be saved by `pickle`.

        :param dirpath: directory path of the saved :class:`ChineseBasicPreprocessor`.
        """
        dirpath = Path(dirpath)
        data_file_path = dirpath.joinpath(self.DATA_FILENAME)

        if data_file_path.exists():
            raise FileExistsError(
                f'{data_file_path} instance exist, fail to save.')
        elif not dirpath.exists():
            dirpath.mkdir()
        self._units = []
        dill.dump(self, open(data_file_path, mode='wb'))

    @classmethod
    def load_preprocessor(cls, dirpath: typing.Union[str, Path]) -> 'mz.DataPack':
        """
        Load the fitted `context`. The reverse function of :meth:`save`.

        :param dirpath: directory path of the saved model.
        :return: a :class:`ChineseBasicPreprocessor` instance.
        """
        dirpath = Path(dirpath)
        data_file_path = dirpath.joinpath(BasePreprocessor.DATA_FILENAME)
        obj = dill.load(open(data_file_path, 'rb'))
        obj._units = ChineseBasicPreprocessor._custom_units()
        return obj
