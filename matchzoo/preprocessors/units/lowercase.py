from .unit import Unit


class Lowercase(Unit):
    """Process unit for text lower case."""

    def __init__(self, split: bool = True):
        self.split = split

    def transform(self, input_: list):
        """
        Convert list of tokens to lower case.

        :param split: return list or str
        :param input_: list of tokens.

        :return tokens: lower-cased list of tokens.
        """
        if self.split:
            return [token.lower() for token in input_]
        else:
            return ''.join([token.lower() for token in input_])
