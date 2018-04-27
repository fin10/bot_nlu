import unittest
from utterance import Utterance


class TestUtterance(unittest.TestCase):

    def test_parse(self):
        tcs = [
            ('busLine', '(강남)[placeName]에서 출발하는 (출근)[busType] 버스 알려줘'),
            ('busLine', '(황골 마을)[placeName] (퇴근)[busType] 버스 알려줘')
        ]

        for domain, tc in tcs:
            utterance = Utterance.parse(domain, tc)
            print(utterance)

    def test_fetch(self):
        result = Utterance.fetch()
        print('\n'.join(list(map(lambda u: str(u), result.values()))))


if __name__ == '__main__':
    unittest.main()
