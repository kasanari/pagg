import csv


class NameGenerator:
    def __init__(self, wordlist) -> None:
        self.wordlist = wordlist

        self.words = []

        with open(wordlist) as csvfile:
            reader = csv.reader(csvfile)
            reader.__next__()
            for row in reader:
                self.words.append(row[0])

        self.current_word = ""

        self.iterator = self.iterate_words()

    def __iter__(self):
        return self

    def iterate_words(self):
        for word in self.words:
            self.current_word = word
            yield word

    def __next__(self):
        return self.iterator.__next__()

    def next(self):
        return self.__next__()

    @property
    def current(self):
        return self.current_word
