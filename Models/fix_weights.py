import torch
import copy

def convert_torchreid_to_ultralytics(source_path, dest_path=None):
    if dest_path is None:
        dest_path = source_path.replace(".pt", "_ready.pt").replace(".pth", "_ready.pt")
        
    print(f"🔄 Elaborazione: {source_path}...")
    
    try:
        # 1. Carica i pesi raw (map_location gestisce CPU/GPU)
        raw_weights = torch.load(source_path, map_location='cpu', weights_only=True)
        
        # Gestione caso in cui il file sia già un dizionario checkpoint
        state_dict = raw_weights
        if 'state_dict' in raw_weights:
            state_dict = raw_weights['state_dict']
        elif 'model' in raw_weights:
            # Se ha già la chiave 'model', controlliamo se è pulita
            print("   ⚠️ Il file sembra già un checkpoint Ultralytics. Verifica se funziona.")
            state_dict = raw_weights['model']
            if not isinstance(state_dict, dict):
                # Caso raro: 'model' è l'intero oggetto nn.Module, non i pesi. 
                # Qui servirebbe estrarre .state_dict(), ma assumiamo sia un dict per ora.
                print("   ⚠️ Attenzione: 'model' non è un dizionario pesi. Potrebbe essere un oggetto modello intero.")
                return

        # 2. Pulizia delle chiavi (Rimuove il prefisso 'module.')
        # Ultralytics OSNet non usa DataParallel wrapper, quindi 'module.' va tolto
        clean_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[7:] # Rimuove i primi 7 caratteri 'module.'
            clean_dict[new_key] = value
            
        print(f"   Chiavi pulite: {len(state_dict)} -> {len(clean_dict)}")

        # 3. Impacchettamento per Ultralytics
        # Ultralytics cerca ckpt['model']. Assegniamo il dizionario pulito a questa chiave.
        ultralytics_ckpt = {'model': clean_dict}

        # 4. Salvataggio
        torch.save(ultralytics_ckpt, dest_path)
        print(f"✅ Salvato correttamente: {dest_path}")
        print("   -> Seleziona QUESTO file nella tua GUI.")

    except Exception as e:
        print(f"❌ Errore critico: {e}")

# --- ESECUZIONE ---
# Inserisci qui i nomi dei file che hai scaricato e rinominato
if __name__ == "__main__":
    # Esempio per i tuoi file (assicurati che siano nella stessa cartella dello script)
    files_to_fix = [
        "osnet_ain_x1_0.pt", 
        "resnet50_msmt17.pt"
    ]
    
    import os
    for f in files_to_fix:
        if os.path.exists(f):
            convert_torchreid_to_ultralytics(f)
        else:
            print(f"⚠️ File non trovato: {f}")