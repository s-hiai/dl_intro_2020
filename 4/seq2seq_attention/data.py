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

    def set_word_id_list(self, word_to_id_dict, set_tags=False):
        for word in self.word_list:
            if word in word_to_id_dict:
                self.word_id_list.append(word_to_id_dict[word])
            else:
                self.word_id_list.append(word_to_id_dict["<unk>"])
        if set_tags:
            self.word_id_list = [word_to_id_dict["<sos>"]] + self.word_id_list + [word_to_id_dict["<eos>"]]

    def set_label_id(self, label_to_id_dict):
        self.label_id = label_to_id_dict[self.label]

    def set_add_vector(self, add_vector):
        self.add_vector = add_vector
        
    def __len__(self):
        return len(self.word_id_list)


class PartOfSentence:
    """
    This class holds a part of sentence (one word id)
    for sentence generation with Seq2Seq model
    """
    def __init__(self, tag_id):
        self.word_id_list = [tag_id]

    def __len__(self):
        return len(self.word_id_list)

    

class SentenceClassificationCorpus:
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



class Seq2SeqCorpus:
    """
    This class holds sentence pair lists of train, dev, test datasets and dictionaries of words and labels
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

        if given_vocab is not None:
            self.word_to_id_dict = given_vocab
            self.id_to_word_dict = {idx:word for word, idx in self.word_to_id_dict.items()}
        
        self.train_sentence_pair_list = []
        self.dev_sentence_pair_list = []
        self.test_sentence_pair_list = []
        
        for line in open(train_file):
            line = line.strip()
            if line == '':
                continue
            text1, text2 = line.split("\t")
            self.train_sentence_pair_list.append((Sentence(text1), Sentence(text2)))
        if given_vocab is None:
            all_train_sentence_list = []
            for sent1, sent2 in self.train_sentence_pair_list:
                all_train_sentence_list.append(sent1)
                all_train_sentence_list.append(sent2)                
            self.set_word_dicts(all_train_sentence_list)
        for sentence1, sentence2 in self.train_sentence_pair_list:
            sentence1.set_word_id_list(self.word_to_id_dict, set_tags=True)
            sentence2.set_word_id_list(self.word_to_id_dict, set_tags=True)

        # dev
        if dev_file is not None:
            for line in open(dev_file):
                line = line.strip()
                if line == '':
                    continue
                text1, text2 = line.split("\t")
                sentence1 = Sentence(text1)
                sentence2 = Sentence(text2)
                sentence1.set_word_id_list(self.word_to_id_dict, set_tags=True)
                sentence2.set_word_id_list(self.word_to_id_dict, set_tags=True)
                self.dev_sentence_pair_list.append((sentence1, sentence2))
                
        # test
        if test_file is not None:
            for line in open(test_file):
                line = line.strip()
                if line == '':
                    continue
                try:
                    text1, text2 = line.split("\t")
                    sentence1 = Sentence(text1)
                    sentence2 = Sentence(text2)
                    sentence1.set_word_id_list(self.word_to_id_dict, set_tags=True)
                    sentence2.set_word_id_list(self.word_to_id_dict, set_tags=True)
                    self.test_sentence_pair_list.append((sentence1, sentence2))
                except:
                    text = line.strip()
                    sentence = Sentence(text, set_tags=True)
                    sentence.set_word_id_list(self.word_to_id_dict)
                    self.test_sentence_pair_list.append((sentence, None))
            

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




        
