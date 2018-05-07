class Vocabulary:

    UNK = '<unk>'

    def __init__(self):
        self.__size = 1
        self.__dict = {self.UNK: 0}
        self.__reverse_dict = {'0': self.UNK}
        self.__freeze = False

    def __len__(self):
        return self.__size

    def freeze(self):
        self.__freeze = True

    def transform(self, text: str):
        if not self.__freeze and text not in self.__dict:
            self.__dict[text] = self.__size
            self.__reverse_dict[str(self.__size)] = text
            self.__size += 1

        return self.__dict[text] if text in self.__dict else self.__dict[self.UNK]

    def restore(self, num: int):
        key = str(num)
        return self.__reverse_dict[key] if key in self.__reverse_dict else self.UNK
