#!/usr/bin/env python3
import os
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL_NAME = "yolo26x-pose.pt"
MIN_MODEL_SIZE_BYTES = 1024 * 1024  # 1 MB safety threshold
DEFAULT_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26x-pose.pt"


def is_valid_model_file(path: Path) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size < MIN_MODEL_SIZE_BYTES:
        return False
    try:
        with path.open("rb") as f:
            head = f.read(128).decode("utf-8", errors="ignore")
        if "git-lfs.github.com/spec/v1" in head:
            return False
    except Exception:
        return False
    return True


def download_with_urllib(url: str, dest: Path) -> None:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response, dest.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)


def download_with_gdown(source: str, dest: Path) -> bool:
    try:
        import gdown  # type: ignore
    except Exception as e:
        print(f"[HERMES] gdown non disponibile ({e}).")
        return False

    url = source
    if not source.lower().startswith("http"):
        url = f"https://drive.google.com/uc?id={source}"

    print(f"[HERMES] Download con gdown da: {url}")
    out = gdown.download(url, str(dest), quiet=False, fuzzy=True)
    return out is not None and dest.exists() and dest.stat().st_size > 0


def try_download(target: Path) -> bool:
    gdrive_source = (
        os.environ.get("HERMES_MODEL_GDRIVE_URL")
        or os.environ.get("HERMES_MODEL_GDRIVE_ID")
    )
    onedrive_url = os.environ.get("HERMES_MODEL_ONEDRIVE_URL")
    direct_url = os.environ.get("HERMES_MODEL_URL")

    sources = []
    if gdrive_source:
        sources.append(("gdown", gdrive_source))
    if onedrive_url:
        sources.append(("urllib", onedrive_url))
    if direct_url:
        sources.append(("urllib", direct_url))
    sources.append(("urllib", DEFAULT_MODEL_URL))

    temp_file = target.with_suffix(target.suffix + ".download")

    for method, source in sources:
        if temp_file.exists():
            temp_file.unlink()
        try:
            if method == "gdown":
                ok = download_with_gdown(source, temp_file)
                if not ok:
                    raise RuntimeError("gdown returned no output file.")
            else:
                print(f"[HERMES] Download via urllib da: {source}")
                download_with_urllib(source, temp_file)

            if not temp_file.exists() or temp_file.stat().st_size == 0:
                raise RuntimeError("Empty file received.")

            temp_file.replace(target)
            return True
        except (HTTPError, URLError) as e:
            print(f"[HERMES] Download fallito ({method}): {e}")
        except Exception as e:
            print(f"[HERMES] Download fallito ({method}): {e}")

    if temp_file.exists():
        temp_file.unlink()
    return False


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / MODEL_NAME

    if is_valid_model_file(model_path):
        print(f"[HERMES] Modello gia presente: {model_path}")
        return 0

    if model_path.exists():
        print(f"[HERMES] Modello trovato ma non valido: {model_path}")
    else:
        print(f"[HERMES] Modello mancante: {model_path}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    if try_download(model_path) and is_valid_model_file(model_path):
        print(f"[HERMES] Download completato: {model_path}")
        return 0

    print("[HERMES] ERRORE: impossibile scaricare il modello.")
    print("[HERMES] Imposta una fonte custom con una di queste variabili:")
    print("         HERMES_MODEL_GDRIVE_URL  (oppure HERMES_MODEL_GDRIVE_ID)")
    print("         HERMES_MODEL_ONEDRIVE_URL")
    print("         HERMES_MODEL_URL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
