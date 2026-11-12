import hashlib
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile


MODELS = (
    (
        "yolox_tiny.onnx",
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx",
        "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
    ),
    (
        "rtmpose_m_ap10k.onnx",
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.zip",
        "1cfd1c86e0d9e5d5f95178bcd95ee9a4e8386a624cd3c57519f27ff58cac7f28",
    ),
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_models(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, url, digest in MODELS:
        target = directory / name
        paths.append(target)
        if target.is_file() and sha256(target) == digest:
            continue

        print(f"Downloading {name} ...", file=sys.stderr, flush=True)
        with TemporaryDirectory(dir=directory) as temporary:
            download = Path(temporary) / "download"
            with urlopen(url, timeout=60) as response, download.open("wb") as output:
                shutil.copyfileobj(response, output)

            model = download
            if url.endswith(".zip"):
                model = Path(temporary) / name
                with ZipFile(download) as archive:
                    member, = (entry for entry in archive.namelist()
                               if Path(entry).name == "end2end.onnx")
                    with archive.open(member) as source, model.open("wb") as output:
                        shutil.copyfileobj(source, output)

            if sha256(model) != digest:
                raise RuntimeError(f"Checksum mismatch for {name}")
            model.replace(target)
    return paths
