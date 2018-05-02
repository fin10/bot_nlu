import re

from konlpy.tag import Twitter
from spans import intrange

from vocabulary import Vocabulary


class Utterance:

    __slot_pattern = re.compile(r'\(([^)]+)\)\[([^\]]+)\]')
    __tokenizer = Twitter()

    def __init__(self, original: str, plain_text: str, tokens: list, labels: list,
                 encoded_tokens: list, encoded_labels: list):
        if len(plain_text) == 0:
            raise ValueError('plain_text should not be empty.')

        if len(tokens) != len(labels):
            raise ValueError('tokens and labels should have same length.')

        self.__original = original
        self.__plain_text = plain_text
        self.__tokens = tokens
        self.__labels = labels
        self.__encoded_tokens = encoded_tokens
        self.__encoded_labels = encoded_labels

    def __str__(self):
        return str({
            'original': self.__original,
            'plain_text': self.__plain_text,
            'tokens': self.__tokens,
            'labels': self.__labels
        })

    def __len__(self):
        return len(self.__tokens)

    @property
    def original(self):
        return self.__original

    @property
    def tokens(self):
        return self.__tokens

    @property
    def labels(self):
        return self.__labels

    @property
    def encoded_tokens(self):
        return self.__encoded_tokens

    @property
    def encoded_labels(self):
        return self.__encoded_labels

    @classmethod
    def parse(cls, utterance: str, text_vocab: Vocabulary, label_vocab: Vocabulary):
        original = utterance
        entities = []
        while True:
            m = cls.__slot_pattern.search(utterance)
            if not m:
                break
            text = m.group(1)
            label = m.group(2)
            start = m.start()
            end = start + len(text)
            entities.append({
                'text': text,
                'label': label,
                'span': intrange(start, end)
            })
            utterance = utterance.replace(m.group(0), text, 1)

        idx = 0
        tokens = []
        labels = []
        for token in cls.__tokenizer.morphs(utterance):
            start = utterance.index(token, idx)
            end = start + len(token)
            span = intrange(start, end)
            tokens.append(token)
            idx = end

            found = False
            for entity in entities:
                if entity['span'].contains(span):
                    label = ('b-' if entity['span'].lower == span.lower else 'i-') + entity['label']
                    labels.append(label)
                    found = True
                    break
            if not found:
                labels.append('o')

        encoded_tokens = [text_vocab.transform(token) for token in tokens]
        encoded_labels = [label_vocab.transform(label) for label in labels]

        return Utterance(original, utterance, tokens, labels, encoded_tokens, encoded_labels)
