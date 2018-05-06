import os
import unittest

from bot import Bot


class TestBot(unittest.TestCase):

    def test_from_local(self):
        bot = Bot.from_local(os.path.join(os.path.dirname(__file__), '../data/busLine'))
        print(bot)

    def test_submit(self):
        bot = Bot.from_local(os.path.join(os.path.dirname(__file__), '../data/busLine'))
        bot.submit()

    def test_fetch(self):
        bot = Bot.fetch('busLine')
        print(bot)

    def test_train(self):
        bot = Bot.fetch('busLine')
        bot.train()

    def test_predict(self):
        bot = Bot.fetch('busLine')
        slots = bot.predict('인계동 가는 퇴근 버스 알려주세요')
        print(slots)


if __name__ == '__main__':
    unittest.main()
