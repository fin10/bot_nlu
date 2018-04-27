import tensorflow as tf
from vocabulary import Vocabulary
from utterance import Utterance


class DatasetGenerator:

    def __init__(self, max_length: int, text_vocab: Vocabulary=Vocabulary(), label_vocab: Vocabulary=Vocabulary()):
        self.__max_length = max_length
        self.__text_vocab = text_vocab
        self.__label_vocab = label_vocab

    def convert(self, utterance: Utterance):
        tokens = list(map(lambda token: self.__text_vocab.transform(token), utterance.tokens()))
        labels = list(map(lambda label: self.__label_vocab.transform(label), utterance.labels()))
        length = len(tokens)

        tokens = [tokens[i] if i < length else 0 for i in range(self.__max_length)]
        labels = [labels[i] if i < length else 0 for i in range(self.__max_length)]
        masks = [1 if i < length else 0 for i in range(self.__max_length)]
        return ({
            'ids': tokens,
            'length': length,
            'mask': masks
        }, labels)

    def generate(self, utterances: list):
        def gen():
            for utterance in utterances:
                yield self.convert(utterance)

        dataset = tf.data.Dataset.from_generator(
            gen,
            (
                {
                    'ids': tf.int32,
                    'length': tf.int32,
                    'mask': tf.int32
                },
                tf.int32
            ),
            (
                {
                    'ids': tf.TensorShape([self.__max_length]),
                    'length': tf.TensorShape([]),
                    'mask': tf.TensorShape([self.__max_length])
                },
                tf.TensorShape([self.__max_length])
            )
        )

        return dataset
