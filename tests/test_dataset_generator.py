import unittest

import tensorflow as tf

from dataset_generator import DatasetGenerator
from utterance import Utterance


class TestDatasetGenerator(unittest.TestCase):

    def test_pre_process(self):
        for (bot_name, utterances) in Utterance.fetch().items():
            ds_generator = DatasetGenerator(10)
            for utterance in utterances:
                print(ds_generator.convert(utterance))

    def test_generate(self):
        for (bot_name, utterances) in Utterance.fetch().items():
            ds_generator = DatasetGenerator(10)
            dataset = ds_generator.generate(utterances)

            with tf.Session() as sess:
                value = dataset.make_one_shot_iterator().get_next()
                print(sess.run(value))


if __name__ == '__main__':
    unittest.main()
