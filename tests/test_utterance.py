import unittest
from utterance import Utterance
from vocabulary import Vocabulary


class TestUtterance(unittest.TestCase):

    def test_parse(self):
        tcs = [
            '(강남)[placeName]에서 출발하는 (출근)[busType] 버스 알려줘',
            '(황골 마을)[placeName] (퇴근)[busType] 버스 알려줘'
        ]

        text_vocab = Vocabulary()
        label_vocab = Vocabulary()

        for tc in tcs:
            utterance = Utterance.parse(tc, text_vocab, label_vocab)
            print(utterance)


if __name__ == '__main__':
    unittest.main()
