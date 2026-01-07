import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [w for w in tokens if w not in stop_words]
    text = " ".join(filtered_tokens)
    return text

def basic_clean(text):
    if not isinstance(text, str): return ""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()