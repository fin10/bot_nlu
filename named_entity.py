class NamedEntity:

    NUM = '_number'

    def __init__(self):
        self.__dict = {}

    def push(self, name: str, words: set):
        self.__dict[name] = list(words)

    def recognize(self, tokens: list):
        result = [[] for _ in range(len(tokens))]
        for i in range(len(tokens)):
            if tokens[i]['text'].isdigit():
                result[i].append(self.NUM)

            for j in range(i + 1, len(tokens) + 1):
                token = ''.join(map(lambda t: t['text'], tokens[i:j]))
                for name, words in self.__dict.items():
                    if token in words:
                        for idx in range(i, j):
                            result[idx].append(('b-' if idx == i else 'i-') + name)
        return result
