#!/usr/bin/env python3
"""Download detector + segmenter weights into weights/.

huggingface.co is unreachable on this server, so HF_ENDPOINT is pointed at
hf-mirror.com before importing huggingface_hub. YOLO-World comes from GitHub
release assets (reachable); SAM2 also from hf-mirror (fbaipublicfiles/Fastly
stalls on large files here).

Downloads:
  weights/owlv2/                      google/owlv2-base-patch16  (OWLv2 detector, fallback)
  weights/yoloworld/yolov8m-worldv2.pt  YOLO-World open-vocab detector (default)
  weights/sam2/sam2.1_hiera_base_plus.pt  SAM2 mask refinement
"""
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

MODEL_ID_OWLV2 = "google/owlv2-base-patch16"
SAM2_REPO = "facebook/sam2.1"
SAM2_CKPT = "sam2.1_hiera_base_plus.pt"
YOLO_URL = ("https://github.com/ultralytics/assets/releases/download/"
            "v8.3.0/yolov8m-worldv2.pt")
YOLO_DEST = os.path.join(WEIGHTS_DIR, "yoloworld", "yolov8m-worldv2.pt")
# CLIP ViT-B/32 text encoder, required by YOLO-World's set_classes (WorldModel
# hardcodes "clip:ViT-B/32"). clip.load() expects it under weights/clip/.
CLIP_URL = ("https://openaipublic.azureedge.net/clip/models/"
            "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/"
            "ViT-B-32.pt")
CLIP_DEST = os.path.join(WEIGHTS_DIR, "clip", "ViT-B-32.pt")


def download_http(url, dest):
    import requests
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {dest} already present", flush=True)
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[start] downloading {url} -> {dest}", flush=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    done = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done/1e6:.1f}/{total/1e6:.1f} MB", end="", flush=True)
    print(f"\n[done] {dest} ({done/1e6:.1f} MB)", flush=True)


def main():
    from huggingface_hub import snapshot_download, hf_hub_download

    # OWLv2 (transformers open-vocab detector)
    owl_dest = os.path.join(WEIGHTS_DIR, "owlv2")
    if not (os.path.exists(os.path.join(owl_dest, "config.json"))
            and os.path.exists(os.path.join(owl_dest, "model.safetensors"))):
        print(f"[start] downloading {MODEL_ID_OWLV2} -> {owl_dest}", flush=True)
        snapshot_download(MODEL_ID_OWLV2, local_dir=owl_dest)
    else:
        print(f"[skip] {owl_dest} already present", flush=True)

    # YOLO-World (open-vocab detector, default)
    download_http(YOLO_URL, YOLO_DEST)

    # CLIP ViT-B/32 (text encoder YOLO-World's set_classes needs)
    download_http(CLIP_URL, CLIP_DEST)

    # SAM2 (mask refinement)
    sam2_dest = os.path.join(WEIGHTS_DIR, "sam2", SAM2_CKPT)
    if not os.path.exists(sam2_dest):
        print(f"[start] downloading {SAM2_REPO}/{SAM2_CKPT} -> {sam2_dest}", flush=True)
        hf_hub_download(SAM2_REPO, SAM2_CKPT, local_dir=os.path.dirname(sam2_dest))
    else:
        print(f"[skip] {sam2_dest} already present", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
