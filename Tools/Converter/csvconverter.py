import pandas as pd
import os
import sys

def convert_csv_to_excel_ita():
    # Richiesta input utente
    input_path = input("Inserisci il percorso del file CSV sorgente: ").strip('"').strip("'")

    if not os.path.exists(input_path):
        print(f"Errore: Il file '{input_path}' non esiste.")
        return

    try:
        # Lettura del CSV (presuppone formato standard USA: virgola come separatore, punto come decimale)
        # Se il file originale ha separatori diversi, modificare 'sep' qui sotto.
        df = pd.read_csv(input_path, sep=',')

        # Generazione nome file output
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_excel_ita{ext}"

        # Scrittura CSV formato ITA
        # sep=';' -> Excel ITA usa il punto e virgola per separare le colonne
        # decimal=',' -> Excel ITA usa la virgola per i decimali
        # encoding='utf-8-sig' -> Garantisce corretta lettura caratteri speciali (accenti) su Excel Windows
        df.to_csv(output_path, sep=';', decimal=',', index=False, encoding='utf-8-sig')

        print(f"Conversione completata. File salvato come: {output_path}")

    except Exception as e:
        print(f"Si è verificato un errore critico: {e}")

if __name__ == "__main__":
    convert_csv_to_excel_ita()