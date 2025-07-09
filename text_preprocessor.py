import nltk
import re
import string
import inflect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# from nltk.stem.porter import PorterStemmer  # optional if using stemming

class TextPreprocessor:
    def __init__(self, remove_stopwords=True):
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')  # for better lemmatizer context

        self.inflect_engine = inflect.engine()
        self.lemmatizer = WordNetLemmatizer()
        # self.stemmer = PorterStemmer()  # optional
        self.stop_words = set(stopwords.words("english")) if remove_stopwords else set()
        self.remove_stopwords = remove_stopwords

    def normalize_case(self, text: str) -> str:
        return text.lower()

    def remove_urls(self, text: str) -> str:
        return re.sub(r'http\S+', '', text)

    def replace_numbers_with_words(self, text: str) -> str:
        return ' '.join([self.inflect_engine.number_to_words(w) if w.isdigit() else w for w in text.split()])

    def remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text: str) -> list:
        return word_tokenize(text)

    def clean_tokens(self, tokens: list) -> list:
        cleaned = [w for w in tokens if w.isalnum()]
        if self.remove_stopwords:
            cleaned = [w for w in cleaned if w not in self.stop_words]
        return cleaned

    def lemmatize_tokens(self, tokens: list) -> list:
        return [self.lemmatizer.lemmatize(w) for w in tokens]

    # def stem_tokens(self, tokens: list) -> list:
    #     return [self.stemmer.stem(w) for w in tokens]  # if using stemmer

    def preprocess_text(self, text: str) -> str:
        text = self.normalize_case(text)
        text = self.remove_urls(text)
        text = self.replace_numbers_with_words(text)
        text = self.remove_punctuation(text)
        tokens = self.tokenize(text)
        tokens = self.clean_tokens(tokens)
        tokens = self.lemmatize_tokens(tokens)  # or stem_tokens
        return ' '.join(tokens)

    def custom_tokenizer(self, text: str) -> list:
        """Use this as tokenizer in TfidfVectorizer"""
        text = self.normalize_case(text)
        text = self.remove_urls(text)
        text = self.replace_numbers_with_words(text)
        text = self.remove_punctuation(text)
        tokens = self.tokenize(text)
        tokens = self.clean_tokens(tokens)
        return self.lemmatize_tokens(tokens)
