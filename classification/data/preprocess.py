import re
import string

import nltk
import stanza
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
            "\U0001f600-\U0001f64f"
            "\U0001f300-\U0001f5ff"
            "\U0001f680-\U0001f6ff"
            "\U0001f1e0-\U0001f1ff"
            "\U00002500-\U00002bef"
            "\U00002702-\U000027b0"
            "\U000024c2-\U0001f251"
            "]+",
            flags=re.UNICODE,
        )

        try:
            stanza.download("ru", verbose=False)
        except Exception as err:
            raise Exception("Stanza download error") from err

        self.nlp = stanza.Pipeline(
            "ru", processors="tokenize,pos,lemma", tokenize_no_ssplit=True, verbose=False
        )

    def _remove_html_tags(self, text: str) -> str:
        return re.sub(r"<.*?>", " ", text)

    def _replace_emoji(self, text: str) -> str:
        text = self.emoji_pattern.sub(" <EMOJI> ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _remove_quotes(self, text: str) -> str:
        return re.sub(r"[«»\"“”„'‘’]", "", text)  # noqa: RUF001

    def _remove_punctuation(self, text: str) -> str:
        translator = str.maketrans("", "", string.punctuation.replace(".", "").replace("%", ""))
        return text.translate(translator)

    # -------------------- TOKENIZE + LEMMA --------------------

    def _tokenize_and_lemm(self, text: str) -> str:
        tokens = word_tokenize(text, language=self.language)
        filtered = [t for t in tokens if t.strip() and t.lower() not in self.all_stopwords]

        text_for_stanza = " ".join(filtered)
        doc = self.nlp(text_for_stanza)
        lemmas = []
        for sent in doc.sentences:
            for word in sent.words:
                if word.text.isnumeric():
                    lemmas.append(word.text)
                else:
                    lemmas.append(word.lemma)

        return " ".join(lemmas)

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
