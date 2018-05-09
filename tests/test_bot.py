import os
import unittest

from bot import Bot


class TestBot(unittest.TestCase):

    def test_from_local(self):
        bot = Bot.from_local(os.path.join(os.path.dirname(__file__), './testBot'))
        print(bot)

    def test_submit(self):
        bot = Bot.from_local(os.path.join(os.path.dirname(__file__), './testBot'))
        bot.submit()

    def test_fetch(self):
        bot = Bot.fetch('testBot')
        print(bot)

    def test_train(self):
        bot = Bot.fetch('testBot')
        result = bot.train()
        print(result)

    def test_predict(self):
        bot = Bot.fetch('testBot')
        slots = bot.predict('show me seafood restaurants')
        print(slots)


if __name__ == '__main__':
    unittest.main()
