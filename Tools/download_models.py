#!/usr/bin/env python3
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MIN_MODEL_SIZE_BYTES = 1024 * 1024  # 1 MB safety threshold

MODEL_SPECS = [
    {
        "filename": "yolo26x-pose.pt",
        "targets": ("yolo26x-pose.pt", "Models/yolo26x-pose.pt"),
        "gdrive_url": "https://drive.google.com/file/d/1h_s5b6h20I8LoCd08WSsPUqd4Q4iwD3R/view?usp=sharing",
        "fallback_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt",
    },
    {
        "filename": "yolo26l-pose.pt",
        "targets": ("Models/yolo26l-pose.pt",),
        "gdrive_url": "https://drive.google.com/file/d/1lTDINGOeeyCLuba0aeNxOiuyJtbU8pUp/view?usp=sharing",
        "fallback_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-pose.pt",
    },
    {
        "filename": "yolo26m-pose.pt",
        "targets": ("Models/yolo26m-pose.pt",),
        "gdrive_url": "https://drive.google.com/file/d/1L-EHt6W-ne9iPPeO1L-kyNvQSRXVccrA/view?usp=sharing",
        "fallback_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-pose.pt",
    },
    {
        "filename": "yolo26s-pose.pt",
        "targets": ("Models/yolo26s-pose.pt",),
        "gdrive_url": "https://drive.google.com/file/d/1AAUY449GzIVmHFBX9tydeRkdMt8L7nBT/view?usp=sharing",
        "fallback_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt",
    },
    {
        "filename": "yolo26n-pose.pt",
        "targets": ("Models/yolo26n-pose.pt",),
        "gdrive_url": "https://drive.google.com/file/d/1kgNQeXfUaVrfYc_rALjaMKBENtQeTbkU/view?usp=sharing",
        "fallback_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt",
    },
    {
        "filename": "osnet_ain_x1_0_ready.pt",
        "targets": ("Models/osnet_ain_x1_0_ready.pt",),
        "gdrive_url": "https://drive.google.com/file/d/1e4X0_4x31gwolJAeCR9Mmmf8wJvxINhx/view?usp=sharing",
        "fallback_url": None,
    },
    {
        "filename": "resnet50_msmt17_ready.pt",
        "targets": ("Models/resnet50_msmt17_ready.pt",),
        "gdrive_url": "https://drive.google.com/file/d/1yu-GpWzZm1DIf83lRKAOsz-undRPhETR/view?usp=sharing",
        "fallback_url": None,
    },
]


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


def extract_drive_file_id(source: str) -> str | None:
    if not source:
        return None
    if source.lower().startswith("http"):
        match = re.search(r"/d/([a-zA-Z0-9_-]+)", source)
        if match:
            return match.group(1)
        match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", source)
        if match:
            return match.group(1)
        return None
    return source


def drive_direct_download_url(source: str) -> str | None:
    file_id = extract_drive_file_id(source)
    if not file_id:
        return None
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def build_sources(model_spec: dict) -> list[tuple[str, str]]:
    filename = model_spec["filename"]
    key = re.sub(r"[^A-Z0-9]", "_", filename.upper())
    env_url = os.environ.get(f"HERMES_{key}_URL")
    env_gdrive = os.environ.get(f"HERMES_{key}_GDRIVE")
    env_onedrive = os.environ.get(f"HERMES_{key}_ONEDRIVE")

    sources = []

    # Per compatibilita', manteniamo le vecchie env vars solo per YOLO principale.
    if filename == "yolo26x-pose.pt":
        legacy_gdrive = (
            os.environ.get("HERMES_MODEL_GDRIVE_URL")
            or os.environ.get("HERMES_MODEL_GDRIVE_ID")
        )
        legacy_onedrive = os.environ.get("HERMES_MODEL_ONEDRIVE_URL")
        legacy_direct = os.environ.get("HERMES_MODEL_URL")
        if legacy_gdrive:
            sources.append(("gdown", legacy_gdrive))
        if legacy_onedrive:
            sources.append(("urllib", legacy_onedrive))
        if legacy_direct:
            sources.append(("urllib", legacy_direct))

    if env_gdrive:
        sources.append(("gdown", env_gdrive))
    if env_onedrive:
        sources.append(("urllib", env_onedrive))
    if env_url:
        sources.append(("urllib", env_url))

    sources.append(("gdown", model_spec["gdrive_url"]))
    fallback_url = model_spec.get("fallback_url")
    if fallback_url:
        sources.append(("urllib", fallback_url))
    return sources


def copy_to_secondary_targets(source_path: Path, targets: list[Path]) -> None:
    for target in targets:
        if target.resolve() == source_path.resolve():
            continue
        if is_valid_model_file(target):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
        print(f"[HERMES] Copiato in: {target}")


def try_download(target: Path, sources: list[tuple[str, str]]) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)

    temp_file = target.with_suffix(target.suffix + ".download")

    for method, source in sources:
        if temp_file.exists():
            temp_file.unlink()
        try:
            if method == "gdown":
                ok = download_with_gdown(source, temp_file)
                if not ok:
                    direct_drive_url = drive_direct_download_url(source)
                    if direct_drive_url:
                        print(f"[HERMES] Tentativo fallback urllib da: {direct_drive_url}")
                        download_with_urllib(direct_drive_url, temp_file)
                    else:
                        raise RuntimeError("gdown returned no output file.")

                if not temp_file.exists() or temp_file.stat().st_size == 0:
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


def ensure_model(project_root: Path, model_spec: dict) -> bool:
    filename = model_spec["filename"]
    targets = [project_root / rel for rel in model_spec["targets"]]
    existing = next((p for p in targets if is_valid_model_file(p)), None)

    if existing:
        print(f"[HERMES] Modello gia presente: {existing}")
        copy_to_secondary_targets(existing, targets)
        return True

    print(f"[HERMES] Modello mancante: {filename}")
    primary_target = targets[0]
    sources = build_sources(model_spec)
    if try_download(primary_target, sources) and is_valid_model_file(primary_target):
        print(f"[HERMES] Download completato: {primary_target}")
        copy_to_secondary_targets(primary_target, targets)
        return True

    print(f"[HERMES] ERRORE: impossibile scaricare {filename}.")
    return False


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    all_ok = True

    for model_spec in MODEL_SPECS:
        ok = ensure_model(project_root, model_spec)
        if not ok:
            all_ok = False

    if all_ok:
        print("[HERMES] Tutti i modelli richiesti sono disponibili.")
        return 0

    print("[HERMES] Alcuni modelli non sono stati scaricati.")
    print("[HERMES] Verifica sharing pubblico su Google Drive (Anyone with the link).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
