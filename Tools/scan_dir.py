import sys
from pathlib import Path

def generate_file_structure(root_dir: str, output_file: str) -> None:
    """
    Scansiona ricorsivamente una directory e scrive il contenuto in un file di testo.
    Utilizza pathlib.Path.walk (Python 3.12+).
    """
    source_path = Path(root_dir)
    
    if not source_path.exists():
        print(f"Errore: La directory '{root_dir}' non esiste.")
        return

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Path.walk() è nativo in Python 3.12
            for dirpath, dirnames, filenames in source_path.walk():
                
                # Calcola il livello di indentazione per la formattazione visiva
                level = len(dirpath.relative_to(source_path).parts)
                indent = '    ' * level
                
                # Scrive la cartella corrente
                f.write(f"{indent}[D] {dirpath.name}/\n")
                
                # Scrive i file contenuti
                sub_indent = '    ' * (level + 1)
                for filename in filenames:
                    f.write(f"{sub_indent}{filename}\n")
                    
        print(f"Scansione completata. Output salvato in: {output_file}")

    except PermissionError as e:
        print(f"Errore di permesso: {e}")
    except Exception as e:
        print(f"Errore imprevisto: {e}")

if __name__ == "__main__":
    # Configurazione
    DIRECTORY_DA_SCANSIONARE = "."  # "." indica la cartella corrente
    NOME_FILE_OUTPUT = "struttura_cartella.txt"

    generate_file_structure(DIRECTORY_DA_SCANSIONARE, NOME_FILE_OUTPUT)