import unittest

from slot_tagger import SlotTagger
from utterance import Utterance


class TestSlotTagger(unittest.TestCase):

    def test_train(self):
        SlotTagger.train('busLine', 10)

    def test_tag(self):
        utterance = Utterance.parse('망포가는 퇴근 버스 알려줘')
        tagger = SlotTagger('busLine', 10)
        tagger.tag(utterance)


if __name__ == '__main__':
    unittest.main()
