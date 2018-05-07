import configparser
import datetime
import json
import os
import pickle
import shutil

from pymongo import MongoClient

from dataset_generator import DatasetGenerator
from named_entity import NamedEntity
from slot_tagger import SlotTagger
from utterance import Utterance
from vocabulary import Vocabulary

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'config.ini'))


class Bot:

    __DATETIME_FORMAT = '%Y%m%d-%H%M%S'

    def __init__(self,
                 name: str,
                 submission_time: datetime,
                 max_text_length: int,
                 max_named_entity_size: int,
                 hyper_params: dict,
                 utterances: set,
                 vocabs: dict,
                 named_entity: NamedEntity
                 ):

        self.__name = name
        self.__submission_time = submission_time
        self.__max_text_length = max_text_length
        self.__max_named_entity_size = max_named_entity_size
        self.__hyper_params = hyper_params
        self.__utterances = utterances
        self.__vocabs = vocabs
        self.__named_entity = named_entity

        model_path = self.__get_model_path()
        self.__slot_tagger = SlotTagger(
            model_path=model_path,
            text_vocab_size=len(self.__vocabs['text']),
            ne_vocab_size=len(self.__vocabs['named_entity']),
            output_size=len(self.__vocabs['label']),
            max_text_length=self.__max_text_length,
            max_named_entity_size=self.__max_named_entity_size,
            hyper_params=self.__hyper_params
        )

    def __str__(self):
        return str({
            'name': self.__name,
            'submission_time': str(self.__submission_time.strftime(self.__DATETIME_FORMAT)),
            'max_text_length': self.__max_text_length,
            'mqx_named_entity_size': self.__max_named_entity_size,
            'hyper_params': self.__hyper_params,
            'utterances': len(self.__utterances)
        })

    def __get_model_path(self):
        return os.path.join(
            os.path.dirname(__file__), './model',
            self.__name,
            str(self.__submission_time.strftime(self.__DATETIME_FORMAT))
        )

    @staticmethod
    def __get_max_text_length(utterances: set):
        max_length = 0
        for utterance in utterances:
            if max_length < len(utterance):
                max_length = len(utterance)
        return max_length * 2

    def train(self):
        model_path = self.__get_model_path()
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        os.makedirs(model_path)

        dataset = DatasetGenerator.generate(self.__max_text_length, self.__max_named_entity_size, self.__utterances)
        dataset = dataset.shuffle(1000).repeat(None).batch(self.__hyper_params['batch_size'])

        self.__slot_tagger.train(dataset, self.__hyper_params['steps'])

    def predict(self, text: str):
        utterance = Utterance.parse(text, self.__vocabs, self.__named_entity)
        dataset = DatasetGenerator.generate(self.__max_text_length, self.__max_named_entity_size, {utterance})
        dataset = dataset.batch(1)

        predictions = self.__slot_tagger.predict(dataset)
        prediction = predictions[0][:len(utterance)]
        labels = list(map(lambda num: self.__vocabs['label'].restore(num), prediction))

        slots = []
        for token, label in zip(utterance.tokens, labels):
            if label.startswith('b-'):
                slots.append({
                    'text': [token],
                    'slot': label.replace('b-', '', 1)
                })
            elif label.startswith('i-'):
                slot = label.replace('i-', '', 1)
                if len(slots) > 0 and slots[-1]['slot'] == slot:
                    slots[-1]['text'].append(token)

        for slot in slots:
            if len(slot['text']) == 1:
                slot['text'] = slot['text'][0]['text']
            else:
                start = slot['text'][0]['span'].lower
                end = slot['text'][-1]['span'].upper
                slot['text'] = utterance.plain_text[start:end]

        return slots

    def submit(self):
        client = MongoClient(CONFIG['default']['mongodb'])
        db = client['bot-nlu']

        for vocab in self.__vocabs.values():
            vocab.freeze()

        bot = {
            'name': self.__name,
            'submission_time': datetime.datetime.utcnow(),
            'max_text_length': self.__max_text_length,
            'max_named_entity_size': self.__max_named_entity_size,
            'hyper_params': self.__hyper_params,
            'utterances': [utterance.original for utterance in self.__utterances],
            'vocabs': pickle.dumps(self.__vocabs),
            'named_entities': pickle.dumps(self.__named_entity),
        }

        db.bot.replace_one({'name': self.__name}, bot, upsert=True)

    @classmethod
    def fetch(cls, bot_name: str):
        client = MongoClient(CONFIG['default']['mongodb'])
        db = client['bot-nlu']
        bot = db.bot.find_one({'name': bot_name})

        vocabs = pickle.loads(bot['vocabs'])
        named_entity = pickle.loads(bot['named_entities'])
        utterances = set(Utterance.parse(utterance, vocabs, named_entity) for utterance in bot['utterances'])

        return Bot(
            name=bot['name'],
            submission_time=bot['submission_time'],
            max_text_length=bot['max_text_length'],
            max_named_entity_size=bot['max_named_entity_size'],
            hyper_params=bot['hyper_params'],
            utterances=utterances,
            vocabs=vocabs,
            named_entity=named_entity
        )

    @classmethod
    def from_local(cls, path: str):
        with open(os.path.join(path, 'config.json'), encoding='utf-8') as fp:
            config = json.loads(fp.read())

        named_entity = NamedEntity()
        for path, _, files in os.walk(path):
            for file in files:
                name, extension = os.path.splitext(file)
                if extension == '.ne':
                    with open(os.path.join(path, file), encoding='utf-8') as fp:
                        words = set(filter(None, map(lambda line: line.replace(' ', '').strip(), fp.readlines())))
                    named_entity.push(name, words)

        vocabs = {
            'text': Vocabulary(),
            'named_entity': Vocabulary(),
            'label': Vocabulary()
        }

        with open(os.path.join(path, 'training'), encoding='utf-8') as fp:
            utterances = set(
                map(lambda line: Utterance.parse(line, vocabs, named_entity),
                    set(filter(None, map(lambda line: line.strip(), fp.readlines()))))
            )

        max_length = cls.__get_max_text_length(utterances)

        return Bot(
            name=config['name'],
            submission_time=datetime.datetime.utcnow(),
            max_text_length=max_length,
            max_named_entity_size=3,
            hyper_params=config['hyper_params'],
            utterances=utterances,
            vocabs=vocabs,
            named_entity=named_entity
        )
