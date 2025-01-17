import string
from typing import List, Union

from nltk import ngrams
from nltk.tokenize import word_tokenize
from vncorenlp import VnCoreNLP

from GraphTranslation.common.languages import Languages
from GraphTranslation.common.ner_labels import *
from objects.graph import SentWord, Sentence, SyllableBasedSentence, SentCombineWord
from GraphTranslation.services.base_service import BaseServiceSingletonWithCache, BaseServiceSingleton
from GraphTranslation.utils.utils import check_number


class NLPCoreService(BaseServiceSingleton):
    def __init__(self):
        super(NLPCoreService, self).__init__()
        self.word_set = set()
        self.max_gram = self.config.max_gram
        self.custom_ner = {}

    @staticmethod
    def check_number(text):
        return check_number(text)

    def add_custom_ner(self, sentence: Sentence) -> Sentence:
        for w in sentence.words:
            if w.original_text in self.custom_ner:
                w.ner_label = self.custom_ner[w.text]
            elif self.check_number(w.original_text):
                w.ner_label = NUM
        return sentence

    @staticmethod
    def word_n_grams(syllables: [SentCombineWord], n) -> List[Union[SentCombineWord, str]]:
        out = list(ngrams(syllables, n=n))
        if not isinstance(syllables[0], str):
            return [SentCombineWord(item) for item in out]
        else:
            return [" ".join(item) for item in out]

    def word_segmentation(self, text):
        raise NotImplementedError("Not Implemented")

    def annotate(self, text) -> Sentence:
        sentence = self.word_segmentation(text)
        sentence = self.add_custom_ner(sentence)
        sentence.update()
        return sentence

    def __call__(self, text):
        return self.make_request(self.annotate, text=text, key=text)


class SrcNLPCoreService(NLPCoreService):
    def __init__(self):
        super(SrcNLPCoreService, self).__init__()
        self.nlpcore_connector = VnCoreNLP(address=self.config.vncorenlp_host, port=self.config.vncorenlp_port,
                                           annotators="wseg,ner")
        self.word_set = self.config.src_word_set
        self.custom_ner = self.config.src_custom_ner

    def word_n_grams(self, words, n):
        if len(words) == 0:
            return []
        if isinstance(words[0], SentWord):
            n_gram_words = list(ngrams(words, n=n))
            new_words = []
            for gram in n_gram_words:
                new_word = SentWord(text=" ".join([w.original_upper for w in gram]),
                                    begin=gram[0].begin, end=gram[-1].end, language=gram[0].language,
                                    pos=",".join([w.pos for w in gram]), ner_label=None)
                new_words.append(new_word)
            return new_words
        else:
            return super().word_n_grams(words, n)

    def map_dictionary(self, text):
        text = text.lower()
        text = " ".join(word_tokenize(text))
        text_ = f" {text} "
        mapped_words = set()
        for n_gram in range(self.max_gram, 0, -1):
            syllables = word_tokenize(text_)
            candidates = self.word_n_grams(syllables, n=n_gram)
            for candidate in candidates:
                if candidate in self.word_set:
                    mapped_words.add(candidate)
                    text_ = text_.replace(f" {candidate} ", "  ")
        return mapped_words

    @staticmethod
    def combine_ner(words: [SentWord]):
        new_words = []
        for w in words:
            if w.ner_label == "O" or w.ner_label is None:
                new_words.append(w)
            elif "B-" in w.ner_label:
                new_words.append(w)
            else:
                pre_word = new_words[-1]
                new_word = SentWord(text=" ".join([pre_word.original_upper, w.original_upper]),
                                    begin=pre_word.begin, end=w.end, language=w.language,
                                    pos=pre_word.pos, ner_label=pre_word.ner_label)
                new_words[-1] = new_word
        return new_words

    def combine_words(self, text, words: [SentWord]):
        mapped_words = self.map_dictionary(text)
        new_words = [w for w in words]
        for n_gram in range(self.max_gram, 0, -1):
            candidates = self.word_n_grams(words, n=n_gram)
            for candidate in candidates:
                if candidate.text in mapped_words:
                    new_words = [w for w in new_words if w not in candidate]
                    new_words.append(candidate)
        new_words.sort(key=lambda w: w.begin)
        for i in range(len(new_words)):
            if i > 0:
                new_words[i].pre = new_words[i-1]
                new_words[i-1].next = new_words[i]
            if i < len(new_words) - 1:
                new_words[i].next = new_words[i+1]
                new_words[i+1].pre = new_words[i]

        return new_words

    def word_segmentation(self, text):
        words = self._annotate(text)
        words = self.combine_ner(words)
        words = self.combine_words(text, words)
        sent = Sentence(words)
        return sent

    def _annotate(self, text):
        text = text.strip()
        for c in string.punctuation:
            text = text.replace(c, f" {c} ")
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        paragraphs = text.split("\n")
        if text[-1] not in "?.:!":
            text += "."
        sentences = []
        for paragraph in paragraphs:
            p_sentences = self.nlpcore_connector.annotate(text=paragraph)["sentences"]
            p_sentences = [sentence + [{"form": "/@", "nerLabel": "O", "posTag": ""}] for sentence in p_sentences]
            p_sentences.append([{"form": "//@", "nerLabel": "O", "posTag": ""}])
            sentences += p_sentences
        # sentences = self.nlpcore_connector.annotate(text=text)["sentences"]
        words = [w for sentence in sentences for w in sentence]
        out = [SentWord(text="@", begin=0, end=1, language=Languages.SRC, pos="")]
        start = 2
        for w in words:
            text = w["form"]
            end = start + len(text)
            word = SentWord(text=w["form"].replace("_", " "), begin=start, end=end, language=Languages.SRC,
                            pos=w.get("posTag", ""), ner_label=w["nerLabel"])
            if len(out) > 0:
                word.pre = out[-1]
                out[-1].next = word
            out.append(word)
            start = end + 1
        return out


class DstNLPCoreService(NLPCoreService):
    def __init__(self):
        super(DstNLPCoreService, self).__init__()
        self.word_set = self.config.dst_word_set
        self.custom_ner = self.config.dst_custom_ner
        self.language = Languages.DST

    def check_number(self, text):
        syllables = word_tokenize(text)
        mask = [False] * len(syllables)
        for n_gram in range(len(syllables), 0, -1):
            for i in range(len(syllables) - n_gram + 1):
                sub_word = " ".join(syllables[i:i+n_gram])
                if sub_word.isnumeric() or self.custom_ner.get(sub_word) == NUM:
                    mask = mask[:i] + [True] * n_gram + mask[i+n_gram:]
        return False not in mask

    def combine_number_tags(self, sentence: Sentence):
        input_text = sentence.text
        new_words = []
        for w in sentence:
            if w.ner_label != NUM:
                new_words.append(w)
            else:
                if len(new_words) == 0 or new_words[-1].ner_label != NUM:
                    new_words.append(w)
                else:
                    pre_word = new_words[-1]
                    new_word = SentWord(text=f"{pre_word.original_text} {w.original_text}", begin=pre_word.begin, end=w.end,
                                        language=pre_word.language, pos=pre_word.pos, ner_label=NUM)
                    new_word.pre = pre_word
                    new_word.next = w.next
                    new_word.pre.next = new_word
                    if new_word.next is not None:
                        new_word.next.pre = new_word
                    new_words[-1] = new_word
        output = Sentence(words=new_words)
        if input_text != output.text:
            raise ValueError(f"{self.__class__.__name__} Cannot do annotation. \n"
                             f"DIFF = {len(output.text) - len(input_text)} \n"
                             f"INPUT = {input_text}\n"
                             f"OUTPUT = {output.text}\n")
        return output

    def word_segmentation(self, text):

        text = " ".join(word_tokenize(text))
        text = f" {text} "
        original_text = text
        text = text.lower()
        input_ = text
        text_ = text
        words = []
        for n_gram in range(self.max_gram, 0, -1):
            syllables = word_tokenize(text_)
            candidates = self.word_n_grams(syllables, n=n_gram)
            for candidate in candidates:
                if candidate in self.word_set and f" {candidate} " in text_:
                    candidate = f" {candidate} "
                    while candidate in text_:
                        begin = text_.find(candidate)
                        end = begin + len(candidate)
                        begin += 1
                        end -= 1
                        words.append(SentWord(text=original_text[begin:end], begin=begin - 1, end=end - 1,
                                              language=self.language, pos=None))
                        text_ = text_[:begin] + " "*(end - begin) + text_[end:]
        not_mapped_words = set(text_.split())
        for candidate in not_mapped_words:
            candidate = f" {candidate} "
            while candidate in text_:
                begin = text_.find(candidate)
                end = begin + len(candidate)
                begin += 1
                end -= 1
                words.append(SentWord(text=original_text[begin:end], begin=begin - 1, end=end - 1,
                                      language=self.language, pos=None))
                text_ = text_[:begin] + " " * (end - begin) + text_[end:]
        words.sort(key=lambda item: item.begin)
        sent = Sentence(words)
        if sent.text != original_text[1:-1]:
            raise ValueError(f"{self.__class__.__name__} Cannot do annotation. \n"
                             f"DIFF = {len(sent.text) - len(original_text[1:-1])} \n"
                             f"INPUT = {input_}\n"
                             f"OUTPUT = {sent.text}\n"
                             f"WORDS = {sent.info}")
        return sent

    def annotate(self, text) -> Sentence:
        sentence = self.word_segmentation(text)
        sentence = self.add_custom_ner(sentence)
        sentence = self.combine_number_tags(sentence)
        sentence.update()
        return sentence


class DictBasedSrcNLPCoreService(DstNLPCoreService):
    def __init__(self):
        super(DictBasedSrcNLPCoreService, self).__init__()
        self.word_set = self.config.src_word_set
        self.custom_ner = self.config.src_custom_ner
        self.language = Languages.SRC


class SyllableBasedDstNLPCoreService(DstNLPCoreService):
    def __init__(self):
        super(SyllableBasedDstNLPCoreService, self).__init__()
        self.word_set = self.config.dst_word_set
        self.custom_ner = self.config.dst_custom_ner
        self.language = Languages.DST
        self.punctuation_set = set(string.punctuation) - set("'")

    def map_dictionary(self, text):
        text = text.lower()
        text = " ".join(word_tokenize(text))
        text_ = f" {text} "
        mapped_words = set()
        for n_gram in range(self.max_gram, 0, -1):
            syllables = word_tokenize(text_)
            # syllables = [SentWord(text=syllable, language=Languages.SRC) for syllable in syllables]
            candidates = self.word_n_grams(syllables, n=n_gram)
            for candidate in candidates:
                if candidate in self.word_set:
                    mapped_words.add(candidate)
                    text_ = text_.replace(f" {candidate} ", "  ")
        # for word in mapped_words:
        #     print(f"{word} : {word in self.word_set}")
        return mapped_words

    def word_n_grams(self, words, n):
        if len(words) == 0:
            return []
        if isinstance(words[0], SentWord):
            n_gram_words = list(ngrams(words, n=n))
            new_words = []
            for gram in n_gram_words:
                # new_word = SentWord(text=" ".join([w.original_upper for w in gram]),
                #                     begin=gram[0].begin, end=gram[-1].end, language=gram[0].language,
                #                     pos=",".join([w.pos for w in gram]), ner_label=None)
                new_word = SentCombineWord(syllables=gram)
                new_words.append(new_word)
            return new_words
        else:
            return super().word_n_grams(words, n)

    def word_tokenize(self, text) -> List[SentWord]:
        text = text.strip()
        if len(text) == 0:
            return []
        if text[-1] not in string.punctuation:
            text += "."
        text = "@ " + text
        for c in "…":
            text = text.replace(c, " ")
        text_ = ""
        for i in range(len(text)):
            if i < len(text) - 1 and text[i] in string.digits and text[i+1] not in string.digits:
                text_ += text[i] + " "
            else:
                text_ += text[i]
        text = text_

        for c in self.punctuation_set:
            text = text.replace(c, f" {c} ")
        words = []
        s = 0
        for i, w in enumerate(text.split()):
            words.append(SentWord(text=w, language=self.language, begin=s, end=s+len(w), pos=None, index=i))
            s += len(w) + 1
        return words

    def word_segmentation(self, text):
        words: [SentWord] = self.word_tokenize(text)
        new_words = []
        for n_gram in range(self.max_gram, 0, -1):
            candidates: List[SentCombineWord] = self.word_n_grams(words, n=n_gram)
            new_words += [c for c in candidates if n_gram == 1 or c.original_text.lower() in self.word_set]
        return SyllableBasedSentence(new_words)

    def combine_number_tags(self, sentence: SyllableBasedSentence) -> SyllableBasedSentence:
        ner_candidates = []
        new_words = []
        for word in sentence.words:
            if word.ner_label == NUM:
                ner_candidates.append(word)
            elif len(ner_candidates) > 0:
                new_word = SentCombineWord([s for w in ner_candidates for s in w.syllables])
                new_words.append(new_word)
                ner_candidates = []
        sentence.build_tree(new_words)
        return sentence


class SyllableBasedSrcNLPCoreService(SyllableBasedDstNLPCoreService):
    def __init__(self):
        super(SyllableBasedSrcNLPCoreService, self).__init__()
        self.word_set = self.config.src_word_set
        self.custom_ner = self.config.src_custom_ner
        self.language = Languages.SRC


class CombinedSrcNLPCoreService(SyllableBasedSrcNLPCoreService):
    def __init__(self):
        super(CombinedSrcNLPCoreService, self).__init__()
        self.nlpcore_connector = VnCoreNLP(address=self.config.vncorenlp_host, port=self.config.vncorenlp_port,
                                           annotators="wseg,ner")

    def get_ner(self, text):
        sentences = self.nlpcore_connector.annotate(text)["sentences"]
        ner_list = []
        for sentence in sentences:
            for word in sentence:
                ner_label = word["nerLabel"]
                if ner_label == "O":
                    continue
                if "B-" in ner_label:
                    ner_list.append(word["form"].replace("_", " "))
                elif "I-" in ner_label:
                    ner_list[-1] += " " + word["form"]
        return set(ner_list)

    def word_segmentation(self, text):
        words: [SentWord] = self.word_tokenize(text)
        ner_set = self.get_ner(text)
        new_words = []
        for n_gram in range(self.max_gram, 0, -1):
            candidates: List[SentCombineWord] = self.word_n_grams(words, n=n_gram)
            for c in candidates:
                if c.original_upper in ner_set:
                    c.ner_label = "ENT"
                    new_words.append(c)
                    continue
                if n_gram == 1 or c.original_text.lower() in self.word_set:
                    new_words.append(c)
            # new_words += [c for c in candidates
            #               if n_gram == 1 or c.original_upper in ner_set or c.original_text.lower() in self.word_set]
        return SyllableBasedSentence(new_words)


class TranslationNLPCoreService(BaseServiceSingleton):
    def __init__(self, is_train=False):
        super(TranslationNLPCoreService, self).__init__()
        self.src_service = SyllableBasedSrcNLPCoreService() if is_train else SrcNLPCoreService()
        self.dst_service = SyllableBasedDstNLPCoreService()
        self.src_dict_based_service = SyllableBasedSrcNLPCoreService()

    def eval(self):
        self.src_service = SrcNLPCoreService()

    def word_segmentation(self, text, language: Languages = Languages.SRC):
        if language == Languages.SRC:
            return self.src_service.annotate(text)
        else:
            return self.dst_service.annotate(text)

    def annotate(self, text, language: Languages = Languages.SRC) -> Sentence:
        if language == Languages.SRC:
            return self.src_service(text)
        else:
            return self.dst_service(text)


if __name__ == "__main__":
    nlpcore_service = TranslationNLPCoreService()
    dst_sentence_ = nlpcore_service.annotate("minh jĭt pơđăm", Languages.DST)
    src_sentence_ = nlpcore_service.word_segmentation("kể từ khi có trạm xá, nơi nhộn nhịp là thành phố Hồ Chí Minh", Languages.SRC)
    # print(dst_sentence_.info)
    # print(src_sentence_.info)
    for w in src_sentence_:
        print(w.id, w.original_text, w.ner_label)
