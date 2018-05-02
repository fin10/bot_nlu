import configparser
import datetime
import json
import os
import shutil
import zlib

from pymongo import MongoClient

from dataset_generator import DatasetGenerator
from slot_tagger import SlotTagger
from utterance import Utterance
from vocabulary import Vocabulary

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'config.ini'))


class Bot:

    __DATETIME_FORMAT = '%Y%m%d-%H%M%S'

    def __init__(self, name: str, creation_time: datetime, max_length: int, hyper_params: dict,
                 utterances: list, text_vocab: Vocabulary, label_vocab: Vocabulary):
        self.__name = name
        self.__creation_time = creation_time
        self.__max_length = max_length
        self.__hyper_params = hyper_params
        self.__utterances = utterances
        self.__text_vocab = text_vocab
        self.__label_vocab = label_vocab

        model_path = self.__get_model_path()
        self.__slot_tagger = SlotTagger(
            model_path, len(self.__text_vocab), len(self.__label_vocab), self.__hyper_params
        ) if os.path.exists(model_path) else None

    def __str__(self):
        return str({
            'name': self.__name,
            'creation_time': str(self.__creation_time.strftime(self.__DATETIME_FORMAT)),
            'max_length': self.__max_length,
            'hyper_params': self.__hyper_params,
            'utterances': len(self.utterances)
        })

    def __get_model_path(self):
        return os.path.join(
            os.path.dirname(__file__), './model',
            self.__name,
            str(self.__creation_time.strftime(self.__DATETIME_FORMAT))
        )

    @staticmethod
    def __get_max_length(utterances: list):
        max_length = 0
        for utterance in utterances:
            if max_length < len(utterance):
                max_length = len(utterance)
        return max_length * 2

    @property
    def name(self):
        return self.__name

    @property
    def max_length(self):
        return self.__max_length

    @property
    def utterances(self):
        return self.__utterances

    def train(self):
        model_path = self.__get_model_path()
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        os.makedirs(model_path)

        dataset = DatasetGenerator.generate(self.__max_length, self.__utterances)
        dataset = dataset.shuffle(1000).repeat(None).batch(100)

        self.__slot_tagger = SlotTagger.train(
            model_path, len(self.__text_vocab), len(self.__label_vocab), self.__hyper_params, dataset
        )

    def predict(self, text: str):
        utterance = Utterance.parse(text, self.__text_vocab, self.__label_vocab)
        dataset = DatasetGenerator.generate(self.__max_length, [utterance])
        dataset = dataset.batch(1)

        predictions = self.__slot_tagger.predict(dataset)
        prediction = predictions[0][:len(utterance)]
        labels = list(map(lambda num: self.__label_vocab.restore(num), prediction))

        print(utterance.tokens)
        print(labels)

    def submit(self):
        client = MongoClient(CONFIG['default']['mongodb'])
        db = client['bot-nlu']

        bot = {
            'name': self.__name,
            'max_length': self.__max_length,
            'hyper_params': self.__hyper_params,
            'text_vocab': zlib.compress(self.__text_vocab.save().encode('utf-8')),
            'label_vocab': zlib.compress(self.__label_vocab.save().encode('utf-8')),
            'utterances': [utterance.original for utterance in self.__utterances],
            'creation_time': datetime.datetime.utcnow()
        }

        db.bot.replace_one({'name': self.__name}, bot, upsert=True)

    @classmethod
    def fetch(cls, bot_name: str):
        client = MongoClient(CONFIG['default']['mongodb'])
        db = client['bot-nlu']
        bot = db.bot.find_one({'name': bot_name})

        text_vocab = Vocabulary(json.loads(zlib.decompress(bot['text_vocab']).decode('utf-8')))
        label_vocab = Vocabulary(json.loads(zlib.decompress(bot['label_vocab']).decode('utf-8')))

        utterances = [Utterance.parse(utterance, text_vocab, label_vocab) for utterance in bot['utterances']]
        max_length = cls.__get_max_length(utterances)

        return Bot(
            bot['name'],
            bot['creation_time'],
            max_length,
            bot['hyper_params'],
            utterances,
            text_vocab,
            label_vocab
        )

    @classmethod
    def from_local(cls, path: str):
        with open(os.path.join(path, 'config.json'), encoding='utf-8') as fp:
            config = json.loads(fp.read())

        text_vocab = Vocabulary()
        label_vocab = Vocabulary()

        with open(os.path.join(path, 'training'), encoding='utf-8') as fp:
            utterances = [Utterance.parse(line.strip(), text_vocab, label_vocab) for line in fp]

        max_length = cls.__get_max_length(utterances)

        return Bot(
            config['name'],
            datetime.datetime.utcnow(),
            max_length,
            config['hyper_params'],
            utterances,
            text_vocab,
            label_vocab
        )
