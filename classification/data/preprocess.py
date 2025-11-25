# classification/data/preprocess.py
import re
import string

import nltk
import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """Class for text preprocessing"""

    def __init__(self, language: str = "russian"):
        self.language = language
        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            nltk.download("stopwords")
            self.stop_words = set(stopwords.words(language))

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        self.all_stopwords = self.stop_words.union()
        self.emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map
            "\U0001f1e0-\U0001f1ff"  # flags
            "\U00002500-\U00002bef"
            "\U00002702-\U000027b0"
            "\U000024c2-\U0001f251"
            "]+",
            flags=re.UNICODE,
        )

        self.morph = pymorphy2.MorphAnalyzer()

    def _remove_html_tags(self, text: str) -> str:
        return re.sub(r"<.*?>", " ", text)

    def _replace_emoji(self, text: str) -> str:
        text = self.emoji_pattern.sub(" <EMOJI> ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _remove_quotes(self, text: str) -> str:
        return re.sub(r"[«»\"“”„'‘’]", "", text)

    def _remove_punctuation(self, text: str) -> str:
        translator = str.maketrans("", "", string.punctuation.replace(".", "").replace("%", ""))
        return text.translate(translator)

    def _tokenize_and_lemm(self, text: str) -> str:
        tokens = word_tokenize(text, language=self.language)
        lemmatized = []
        for token in tokens:
            new_token = token.strip()
            if not new_token:
                continue
            if new_token.lower() in self.all_stopwords:
                continue
            if new_token.isnumeric():
                lemmatized.append(new_token)
                continue
            lemma = self.morph.parse(new_token)[0].normal_form
            lemmatized.append(lemma)
        return " ".join(lemmatized)

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = self._remove_html_tags(text)
        text = self._remove_quotes(text)
        text = self._replace_emoji(text)
        text = self._remove_punctuation(text)
        text = text.strip()
        return self._tokenize_and_lemm(text)
