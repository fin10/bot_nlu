import unittest

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


if __name__ == '__main__':
    unittest.main()

