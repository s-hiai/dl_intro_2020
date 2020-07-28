import MeCab
tagger = MeCab.Tagger("-Owakati")
tagger.parse("")

class Sentence:
    """
    This class holds 
    sentence text, the words and the ids in text sentence label, the label and the id
    """
    def __init__(self, text, label=None):
        self.text = text
        self.word_list = self.parse(text)
        self.label = label
        self.word_id_list = []
        self.label_id = -1
        self.add_vector = None

    def parse(self, text):
        """
        sentence parsing (word split)
        :param text: sentence text
        :return word_list: word list in sentence text
        """
        text = tagger.parse(text).strip()
        word_list = text.split()
        return word_list

    def set_word_id_list(self, word_to_id_dict):
        for word in self.word_list:
            if word in word_to_id_dict:
                self.word_id_list.append(word_to_id_dict[word])
            else:
                self.word_id_list.append(word_to_id_dict["<unk>"])

    def set_label_id(self, label_to_id_dict):
        self.label_id = label_to_id_dict[self.label]

    def set_add_vector(self, add_vector):
        self.add_vector = add_vector
        
    def __len__(self):
        return len(self.word_id_list)
    

class Corpus:
    """
    This class holds sentence lists of train, dev, test datasets and dictionaries of words and labels
    """
    def __init__(
            self,
            train_file,
            dev_file=None,
            test_file=None,
            given_vocab=None
    ):
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.word_to_id_dict = None
        self.id_to_word_dict = None
        self.label_to_id_dict = None
        self.id_to_label_dict = None

        if given_vocab is not None:
            self.word_to_id_dict = given_vocab
            self.id_to_word_dict = {idx:word for word, idx in self.word_to_id_dict.items()}
        
        self.train_sentence_list = []
        self.dev_sentence_list = []
        self.test_sentence_list = []
        
        for line in open(train_file):
            line = line.strip()
            if line == '':
                continue
            split_line = line.split("\t")
            label, text = split_line[0], "\t".join(split_line[1:])
            self.train_sentence_list.append(Sentence(text, label))
        if given_vocab is None:
            self.set_word_dicts(self.train_sentence_list)
        self.set_label_dicts(self.train_sentence_list)
        for sentence in self.train_sentence_list:
            sentence.set_word_id_list(self.word_to_id_dict)
            sentence.set_label_id(self.label_to_id_dict)

        # dev
        if dev_file is not None:
            for line in open(dev_file):
                line = line.strip()
                if line == '':
                    continue
                split_line = line.split("\t")
                label, text = split_line[0], "\t".join(split_line[1:])
                sentence = Sentence(text, label)
                sentence.set_word_id_list(self.word_to_id_dict)
                sentence.set_label_id(self.label_to_id_dict)
                self.dev_sentence_list.append(sentence)

                
        # test
        if test_file is not None:
            for line in open(test_file):
                line = line.strip()
                if line == '':
                    continue
                try:
                    split_line = line.split("\t")
                    label, text = split_line[0], "\t".join(split_line[1:])
                    sentence = Sentence(text, label)
                    sentence.set_label_id(self.label_to_id_dict)
                except:
                    text = line.strip()
                    sentence = Sentence(text)
                sentence.set_word_id_list(self.word_to_id_dict)
                self.test_sentence_list.append(sentence)
            

    def set_word_dicts(self, sentence_list):
        """
        create word-to-id dict and id-to-word dict
        :param sentence_list: sentence list for training
        """
        word_to_id_dict = {"<pad>":0, "<unk>":1, "<sos>":2, "<eos>":3}
        for sentence in sentence_list:
            for word in sentence.word_list:
                if word not in word_to_id_dict:
                    word_to_id_dict[word] = len(word_to_id_dict)
        self.word_to_id_dict = word_to_id_dict
        id_to_word_dict = {}
        for word, idx in word_to_id_dict.items():
            id_to_word_dict[idx] = word
        self.id_to_word_dict = id_to_word_dict

        
    def set_label_dicts(self, sentence_list):
        """
        create label-to-id dict and id-to-label dict
        :param sentence_list: sentence list for training
        """
        label_to_id_dict = {}
        for sentence in sentence_list:
            if sentence.label not in label_to_id_dict:
                label_to_id_dict[sentence.label] = len(label_to_id_dict)
        self.label_to_id_dict = label_to_id_dict
        id_to_label_dict = {}
        for label, idx in label_to_id_dict.items():
            id_to_label_dict[idx] = label
        self.id_to_label_dict = id_to_label_dict


        
class DataSet:
    """
    データセットクラス
    """
    def __init__(self, text_list, label_list=None, vocab=None, label_dict=None):
        if label_list is None:
            self.sentence_list = [Sentence(text, None) for text in text_list]
        else:
            self.sentence_list = [Sentence(text, label) for text, label in zip(text_list, label_list)]
        if vocab is None:
            self.vocab = self.set_word_dicts(self.sentence_list)
        else:
            self.vocab = vocab
        if label_dict is None:
            self.label_dict = {}
            for label_id, label_name in enumerate(set([sentence.label_name for sentence in self.sentence_list])):
                self.label_dict[label_name] = label_id
        else:
            self.label_dict = label_dict
        for sentence in self.sentence_list:
            sentence.set_word_id_list(self.vocab)
        if label_list is not None:
            for sentence in self.sentence_list:
                sentence.set_label(self.label_dict)
            
            
    def __len__(self):
        return len(self.sentence_list)


    def set_word_dicts(self, sentence_list):
        """
        create word-to-id dict and id-to-word dict
        :param sentence_list: sentence list for training
        """
        word_to_id_dict = {"<pad>":0, "<unk>":1}
        for sentence in sentence_list:
            for word in sentence.word_list:
                if word not in word_to_id_dict:
                    word_to_id_dict[word] = len(word_to_id_dict)
        self.word_to_id_dict = word_to_id_dict
        id_to_word_dict = {0:"<pad>", 1:"<unk>"}
        for word, idx in word_to_id_dict.items():
            id_to_word_dict[idx] = word
        self.id_to_word_dict = id_to_word_dict


    
    def make_vocab(self):
        """
        語彙リストの作成
        """
        all_words = {}
        for sentence in self.sentence_list:
            all_words.extend(sentence.word_list)
        return list(set(all_words))



        
