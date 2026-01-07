import os
import json
import pandas as pd
import re
from src.text_processing import basic_clean
from src.data_loading import load_journals_data

def build_master_csv(input_path='data/raw', output_path='data/processed/master_dataset.csv'):
    """Usa load_journals_data para crear el dataset unificado."""
    print(f"Leyendo datos desde {input_path}...")
    
    # Llamada a tu función
    df = load_journals_data(input_path)
    
    if df.empty:
        print("No se encontraron datos.")
        return

    # Aplicar limpieza ligera a las columnas principales
    print("Aplicando preprocesamiento común...")
    df['title'] = df['title'].apply(basic_clean)
    df['abstract'] = df['abstract'].apply(basic_clean)
    
    # Asegurar que keywords sea un string (unir si es lista)
    df['keywords'] = df['keywords'].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else basic_clean(str(x))
    )

    # Crear la columna combinada (mínimo preprocesado)
    df['full_text'] = (df['title'] + " " + df['abstract'] + " " + df['keywords']).str.strip()

    # Guardar el CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"¡Éxito! Master Dataset guardado en: {output_path}")
    print(f"Total de registros: {len(df)}")
    print(f"Columnas disponibles: {df.columns.tolist()}")

if __name__ == "__main__":
    # Ajusta 'Journals' a la ruta donde tengas tus carpetas
    build_master_csv(input_path='data/raw')