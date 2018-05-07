import re

import pyokt
from spans import intrange

from named_entity import NamedEntity


class Utterance:

    __slot_pattern = re.compile(r'\(([^)]+)\)\[([^\]]+)\]')

    def __init__(self, original: str, plain_text: str, tokens: list,
                 encoded_tokens: list, encoded_entities: list, encoded_labels: list):
        if len(plain_text) == 0:
            raise ValueError('plain_text should not be empty.')

        if len(encoded_tokens) != len(encoded_entities) != len(encoded_labels):
            raise ValueError('tokens, labels and entities should have same length.')

        self.__original = original
        self.__plain_text = plain_text
        self.__tokens = tokens
        self.__encoded_tokens = encoded_tokens
        self.__encoded_entities = encoded_entities
        self.__encoded_labels = encoded_labels

    def __str__(self):
        return str({
            'original': self.__original,
            'plain_text': self.__plain_text,
            'tokens': self.__tokens
        })

    def __len__(self):
        return len(self.__tokens)

    @property
    def plain_text(self):
        return self.__plain_text

    @property
    def original(self):
        return self.__original

    @property
    def tokens(self):
        return self.__tokens

    @property
    def encoded_tokens(self):
        return self.__encoded_tokens

    @property
    def encoded_entities(self):
        return self.__encoded_entities

    @property
    def encoded_labels(self):
        return self.__encoded_labels

    @classmethod
    def parse(cls, utterance: str, vocabs: dict, named_entity: NamedEntity):
        original = utterance
        items = []
        while True:
            m = cls.__slot_pattern.search(utterance)
            if not m:
                break
            text = m.group(1)
            label = m.group(2)
            start = m.start()
            end = start + len(text)
            items.append({
                'text': text,
                'label': label,
                'span': intrange(start, end)
            })
            utterance = utterance.replace(m.group(0), text, 1)

        tokens = []
        labels = []
        for token in pyokt.tokenize(utterance):
            if token.text.isspace():
                continue

            span = intrange(token.offset, token.offset + token.length)
            tokens.append({
                'text': token.text,
                'span': span
            })

            found = False
            for item in items:
                if item['span'].contains(span):
                    label = ('b-' if item['span'].lower == span.lower else 'i-') + item['label']
                    labels.append(label)
                    found = True
                    break
            if not found:
                labels.append('o')

        encoded_tokens = [vocabs['text'].transform(token['text']) for token in tokens]
        encoded_labels = [vocabs['label'].transform(label) for label in labels]

        entities = named_entity.recognize(tokens)
        encoded_entities = [
            [vocabs['named_entity'].transform(unit) for unit in entity] for entity in entities
        ]

        return Utterance(
            original, utterance, tokens,
            encoded_tokens, encoded_entities, encoded_labels
        )
