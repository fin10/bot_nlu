import unittest
import json
from vocabulary import Vocabulary


class TestVocabulary(unittest.TestCase):

    def test_unk(self):
        vocab = Vocabulary()
        self.assertEqual(0, vocab.transform(Vocabulary.UNK))
        self.assertEqual(Vocabulary.UNK, vocab.restore(0))

    def test_transform(self):
        vocab = Vocabulary()
        num = vocab.transform('test')
        text = vocab.restore(num)
        self.assertEqual(2, len(vocab))
        self.assertEqual('test', text)

    def test_save(self):
        vocab = Vocabulary()
        expected = [vocab.transform('test'), vocab.transform('hello')]
        saved = vocab.save()

        new_vocab = Vocabulary(json.loads(saved))
        actual = [new_vocab.transform('test'), new_vocab.transform('hello')]

        self.assertListEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()

