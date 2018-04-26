import json


class Vocabulary:

    UNK = '<unk>'

    def __init__(self, data=None):
        if data:
            self.__size = data['size']
            self.__dict = data['dict']
            self.__reverse_dict = data['reverse_dict']
            self.__freeze = True
        else:
            self.__size = 1
            self.__dict = {self.UNK: 0}
            self.__reverse_dict = {0: self.UNK}
            self.__freeze = False

    def __len__(self):
        return self.__size

    def transform(self, text: str):
        if not self.__freeze and text not in self.__dict:
            self.__size += 1
            self.__dict[text] = self.__size
            self.__reverse_dict[self.__size] = text

        return self.__dict[text] if text in self.__dict else self.__dict[self.UNK]

    def restore(self, num: int):
        return self.__reverse_dict[num] if num in self.__reverse_dict else self.UNK

    def save(self):
        return json.dumps({
            'size': self.__size,
            'dict': self.__dict,
            'reverse_dict': self.__reverse_dict
        })
