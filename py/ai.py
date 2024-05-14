"""
This module provides some simple tools for AI education.  We deliberately do not use the
more sophisticated packages like TensorFlow etc., but try to stick to the absolute basics.

This is a draft and not for public distribution.  Many parts of this module are unfinished
and work in progress.

(c) 2024, Tobias Kohn, Dennis Komm
"""
import math, os


# IMAGE-RECOGNITION AND MANIPULATION

class Image:
    """
    Represents an MNIST-image (28x28).
    """

    def __init__(self, data=None, label=None, index=None):
        if data:
            self.data = bytearray(data)
            self.norm = math.sqrt(sum(a * a for a in self.data))
        else:
            self.data = bytearray(28 * 28)
            self.norm = 0
        self.label = label
        self.index = index

    def __getitem__(self, item):
        i, j = item
        return self.data[28 * i + j]

    def __setitem__(self, key, value):
        i, j = key
        self.data[28 * i + j] = value
        self.norm = math.sqrt(sum(a * a for a in self.data))

    def __add__(self, other):
        """
        Adds an image as a difference and returns a new image.
        """
        data = bytearray(self.data)
        if other.label and other.label < 0:
            for i, (v, d) in enumerate(zip(data, other.data)):
                data[i] = min(max(0, v + (d - 0x80)), 0xFF)
        else:
            for i, (v, d) in enumerate(zip(data, other.data)):
                data[i] = min(max(0, v + d), 0xFF)
        return Image(data, label=self.label)
    
    def __sub__(self, other):
        """
        Returns half the difference between two images.  The half is to make the 
        difference fit into a byte.
        """
        data = bytearray(self.data)
        for i, (v, d) in enumerate(zip(data, other.data)):
            data[i] = min(max(0, (v - d) // 2 + 0x80), 0xFF)
        label = -abs(self.label - other.label) if self.label and other.label and self.label != other.label else -1
        return Image(data, label=label)
        
    def __mul__(self, other):
        data = bytearray(self.data)
        if self.label and self.label < 0:
            for i, v in enumerate(data):
                data[i] = min(255, max(0, int((v - 0x80) * other) + 0x80))
        else:
            for i, v in enumerate(data):
                data[i] = min(255, max(0, int(v * other)))
        return Image(data, label=self.label)

    def __floordiv__(self, other):
        data = bytearray(self.data)
        if self.label and self.label < 0:
            for i, v in enumerate(data):
                data[i] = (v - 0x80) // other + 0x80
        else:
            for i, v in enumerate(data):
                data[i] = v // other
        return Image(data, label=self.label)

    def __truediv__(self, other):
        data = bytearray(self.data)
        if self.label and self.label < 0:
            for i, v in enumerate(data):
                data[i] = int((v - 0x80) / other) + 0x80
        else:
            for i, v in enumerate(data):
                data[i] = int(v / other)
        return Image(data, label=self.label)

    def dot(self, other) -> float:
        """
        Computes the dot-product of this image and another one.
        """
        result = sum( a*b for a, b in zip(self.data, other.data) )
        return result / (self.norm * other.norm)

    def draw(self) -> None:
        """
        Draws the image to the console using ASCII art.
        """
        if self.label:
            print("LABEL:", self.label)
        print('\n'.join(self.toStringList()))
        
    def edges(self):
        """
        Detects the edges by line scan and returns an image containing those edges.
        """
        data = []
        for i in range(28):
            line = self.data[28*i:28*(i+1)]
            for j in range(27):
                data.append(abs(line[j+1] - line[j]))
            data.append(0)
        return Image(data, label=self.label)
    
    def gauss_conv(self):
        data = []
        data.extend([0] * 28)
        for i in range(1, 27):
            data.append(0)
            line_a = self.data[28*(i-1):28*i]
            line   = self.data[28*i:28*(i+1)]
            line_b = self.data[28*(i+1):28*(i+2)]
            for j in range(1, 27):
                d = 4 * line[j] - (line[j-1] + line[j+1]) - (line_a[j] + line_b[j])
                data.append(max(0, min(d, 255)))
            data.append(0)
        data.extend([0] * 28)
        return Image(data, label=self.label)
        
    def toStringList(self) -> list:
        """
        Draws the image to the console using ASCII art.
        """
        scale = " .:-=+*#%%@@@@"
        result = []
        current = []
        for i, d in enumerate(self.data):
            if i % 28 == 0 and i > 0:
                result.append(''.join(current))
                current = []
            current.append(scale[12 * d * d // (255 * 255)])
        result.append(''.join(current))
        return result


def load_images_from_files(data_file: str, label_file: str, verbose: bool = True) -> list:
    images = []
    with open(data_file, 'rb') as f:
        data = f.read()
    with open(label_file, 'rb') as f:
        labels = f.read()
    i = 16
    j = 8
    idx = 0
    if verbose: print("Loading images", end='')
    while i < len(data) and j < len(labels):
        img = Image( data[i:i+28*28], labels[j], index=idx )
        images.append(img)
        i += 28*28
        j += 1
        idx += 1
        if verbose and idx % 1000 == 0: print('.', end='', flush=True)
    if verbose: print("done")
    return images


def load_mnist_images(dataset: str = 'train', verbose: bool = True) -> list:
    """
    Loads the mnist datasets and returns a list with the respective images.

    The parameter is set to either 'train' or 'test' to get the training or test-data, respectively.
    """
    if dataset.lower() == 'test':
        return load_images_from_files('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', verbose)
    else:
        return load_images_from_files('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', verbose)


def draw_side_by_side(*images):
    """
    Draws the given images side by side.
    """
    if len(images) == 1 and isinstance(images[0], (list, tuple)):
        images = images[0]
    texts = [img.toStringList() for img in images]
    for lines in zip(*texts):
        print(' | '.join(lines))


# TOKENIZATION FOR CHAT-BOTS AND LLMs

def tokenize(phrase: str, symbols: str = ".:,;!?-") -> list:
    """
    Splits a given input string into a sequence of tokens (words, numbers and punctuation marks).
    """
    result = []
    j = 0
    while j < len(phrase):
        while j < len(phrase) and phrase[j].isspace(): j += 1
        if j == len(phrase): break
        c = phrase[j]
        if (f := [lambda s: s.isalpha() or s in "'_", str.isdigit][c.isdigit()]) and f(c):
            i = j
            while j < len(phrase) and f(phrase[j]):
                j += 1
            result.append(phrase[i:j])
        elif phrase[j:j+3] == '...':
            result.append('...')
            j += 3
        else:
            if c in symbols: result.append(c)
            j += 1
    return result


def normalize(tokens: list) -> list:
    """
    Makes sure all tokens are lower case and splits the input into individual sentences.  
    The returned structure is a list of sentences (each one a list on its own).  Note that
    the only punctuation marks that are kept are question marks.
    """
    result = [[]]
    for token in tokens:
        if token in '.!?':
            if token == '?': result[-1].append(token)
            result.append([])
        elif token not in '.:,;?!-':
            result[-1].append(token.lower())
    return [s for s in result if s]


# LOAD TEXTS FOR FURTHER ANALYSIS

_corrections = {
    'pechmaria': 'pechmarie',
    'prinzessinen': 'prinzessinnen',
    'mondenschein': 'mondschein',
    'kirchthürme': 'kirchtürme',
    'gieng': 'ging',
    'ward': 'wurde',
    'gethan': 'getan',
    'rathschläge': 'ratschläge',
    'rathes': 'rates',
    'mausetodt': 'mausetot',
    'todt': 'tot',
    'hausthüre': 'haustüre',
    'hinterthüre': 'hintertüre',
    'thüre': 'türe',
    'thoren': 'toren',
    'thor': 'tor',
    'liebenswerther': 'liebenswerter',
    'muthmassungen': 'muthmassungen',
    'muth': 'mut',
    'erröthend': 'errötend',
    'übermüthig': 'übermütig',
}

def _normalize_word(word):
    if word == '':
        return word
    elif word[-1] in '.,;:!?':
        return _normalize_word(word[:-1]) + word[-1]
    else:
        word2 = _corrections.get(word.lower(), word)
        if word2 != word and word[0].isupper():
            return word2.title()
        else:
            return word2

def _handle_line(filename: str, line: str):
    if line.startswith("[Illustration]"):
        return ''
    orig_line = line
    line = [_normalize_word(s) for s in line.translate(
      str.maketrans('\t\'', '  ', '*_()-–—\n\r\"„“»«<>|+')
    ).replace('ß', 'ss').replace(', und ', ' und ').split(' ') if s != '']
    return ' '.join(line)
    
def _handle_file(filename: str, verbose: bool):
    if verbose: print(filename[filename.rindex('/')+1:], end='')
    result = []
    i = 0
    with open(filename) as f:
        paragraph = []
        for line in f:
            line = line.strip()
            if line == '' and paragraph:
                result.append(_handle_line(filename, ' '.join(paragraph)))
                paragraph = []
                i += 1
                if verbose and i % 100 == 0:
                    print('.', end='', flush=True)
            else:
                paragraph.append(line)
        if paragraph:
            result.append(_handle_line(filename, ' '.join(paragraph)))
    if verbose: print(f"({i} paragraphs)", flush=True)
    return '\n'.join(result)

def load_text_files(path, verbose: bool = True):
    result = []
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.name.endswith('.txt'):
                result.append( _handle_file(f"{path}/" + entry.name, verbose) )
    return result

