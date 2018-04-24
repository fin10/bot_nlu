import unittest
from utterance import Utterance


class TestUtteranceParser(unittest.TestCase):

    def test_parse(self):
        utterance = Utterance.parse('(황골 마을)[placeName] (퇴근)[busType] 버스 알려줘')
        print(utterance)


if __name__ == '__main__':
    unittest.main()
