import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class DatasetFactory:
    def __init__(self, master_path='data/processed/master_dataset.csv'):
        if not os.path.exists(master_path):
            raise FileNotFoundError(f"Master Dataset not found at {master_path}. Please run the builder first.")
        self.df = pd.read_csv(master_path)
        self.stop_words = set(stopwords.words('english'))

    def _clean_for_ml(self, text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        cleaned = [w for w in tokens if w not in self.stop_words]
        return " ".join(cleaned)

    def _clean_for_dl(self, text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s?.,!]', '', text)
        return " ".join(text.split())

    def create_ml_dataset(self, experiment_name='classic_ml_default'):
        print(f"Generating ML dataset for experiment: {experiment_name}...")
        
        df_ml = self.df.copy()
        df_ml['processed_text'] = df_ml['full_text'].apply(self._clean_for_ml)
        
        path = f'data/experiments/{experiment_name}'
        os.makedirs(path, exist_ok=True)
        
        output_file = f'{path}/dataset_ml.csv'
        df_ml[['journal', 'processed_text']].to_csv(output_file, index=False)
        print(f"Dataset ML guardado en {output_file}")
        return df_ml

    def create_dl_dataset(self, experiment_name='conexionista_dl_default'):
        print(f"Generating DL dataset for experiment: {experiment_name}...")
        
        df_dl = self.df.copy()
        df_dl['processed_text'] = df_dl['abstract'].apply(self._clean_for_ml)
        df_dl['label_idx'] = pd.Categorical(df_dl['journal']).codes

        mapping = dict(enumerate(pd.Categorical(df_dl['journal']).categories))

        path = f'data/experiments/{experiment_name}'
        os.makedirs(path, exist_ok=True)
        df_dl[['label_idx', 'processed_text']].to_csv(f'{path}/dataset_dl.csv', index=False)
        with open(f'{path}/label_mapping.json', 'w') as f:
            json.dump(mapping, f)
        print(f"Dataset DL and mapping saved in {path}")
        return df_dl