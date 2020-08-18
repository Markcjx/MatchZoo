import string

from .unit import Unit


class ChinesePuncRemoval(Unit):
    """Process unit for remove punctuations."""
    def __init__(self,split:bool = True):
        self.split = split
        self.punctuation = string.punctuation + '，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥'

    def transform(self, input_: list) -> list:
        """
        Remove punctuations from list of tokens.

        :param input_: list of toekns.

        :return rv: tokens  without punctuation.
        """
        table = str.maketrans({key: None for key in self.punctuation})
        if self.split:
            result = [item.translate(table) for item in input_]
        else:
            result = ''.join([item.translate(table) for item in input_])
        return result
