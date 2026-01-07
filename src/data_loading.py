import os
import json
import pandas as pd

def load_journals_data(base_path):
    all_articles = []
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        if os.path.isdir(folder_path):
            journal_label = folder_name
            
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            articles_list = json.load(f)
                            if not isinstance(articles_list, list):
                                articles_list = [articles_list]
                            
                            for article in articles_list:
                                article['journal'] = journal_label
                                all_articles.append(article)
                        except json.JSONDecodeError:
                            continue
                            
    return pd.DataFrame(all_articles)