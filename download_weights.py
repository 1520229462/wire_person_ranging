#!/usr/bin/env python3
"""Download detector + segmenter weights into weights/ (+ optional VGGT depth cache).

YOLO-World / CLIP / SAM2 直连官方源（GitHub release 资产 / OpenAI CDN /
huggingface.co）。本开发服务器无法直连 huggingface.co，请在这台机器上设置
环境变量 `WPR_HF_MIRROR=1` 走 hf-mirror.com 镜像。所有下载按 weights/README.md
中的 md5/sha256 校验。

TTPLA checkpoint 与 VGGT 深度缓存为**私有分发**（见 weights/README.md）：
本脚本打印手动放置指引并对已放置的文件校验哈希。

下载内容:
  weights/yoloworld/yolov8m-worldv2.pt   YOLO-World 开集检测器（默认检测器）
  weights/sam2/sam2.1_hiera_base_plus.pt SAM2 掩膜精化
  weights/clip/ViT-B-32.pt               CLIP 文本编码器（YOLO-World set_classes 需要）
  weights/ttpla/yolo11s_seg_1280.pt      TTPLA 导线分割（--wire_mode per_frame 必需）
  --with-owlv2                           OWLv2 开集检测兜底（可选，需 requirements-owlv2.txt）
  --with-vggt-depth                      data/vggt_depth/ 28 帧深度缓存（可选，
                                         缺失时人分支自动退化为最近簇）

用法:
  python download_weights.py                 # 核心权重（含 TTPLA）
  python download_weights.py --with-owlv2 --with-vggt-depth
"""
import argparse
import hashlib
import os
import sys
import zipfile

# 本服务器网络变通：huggingface.co 主站 443 超时，只有显式要求才走镜像；
# 第三方正常网络无需设置，直连官方更快。
if os.environ.get("WPR_HF_MIRROR") == "1" and "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

MODEL_ID_OWLV2 = "google/owlv2-base-patch16"
SAM2_REPO = "facebook/sam2.1"
SAM2_CKPT = "sam2.1_hiera_base_plus.pt"

YOLO_URL = ("https://github.com/ultralytics/assets/releases/download/"
            "v8.3.0/yolov8m-worldv2.pt")
YOLO_DEST = os.path.join(WEIGHTS_DIR, "yoloworld", "yolov8m-worldv2.pt")
YOLO_MD5 = "63c634cc53ab480afb320250d298efc1"

# CLIP ViT-B/32 text encoder, required by YOLO-World's set_classes (WorldModel
# hardcodes "clip:ViT-B/32"). clip.load() expects it under weights/clip/.
CLIP_URL = ("https://openaipublic.azureedge.net/clip/models/"
            "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/"
            "ViT-B-32.pt")
CLIP_DEST = os.path.join(WEIGHTS_DIR, "clip", "ViT-B-32.pt")
CLIP_MD5 = "3ba34e387b24dfe590eeb1ae6a8a122b"

SAM2_MD5 = "ec7bd7d23d280d5e3cfa45984c02eda5"

# TTPLA 微调 checkpoint（默认 per_frame 导线模式必需）。
# 分发方式为**私有分发**（见 weights/README.md）：由原作者私下提供文件，
# 放置后本脚本校验 md5/sha256。若日后设置公开下载地址，用环境变量
# WPR_TTPLA_URL 指向它，脚本即自动下载并校验。
TTPLA_URL = os.environ.get("WPR_TTPLA_URL", "")
TTPLA_DEST = os.path.join(WEIGHTS_DIR, "ttpla", "yolo11s_seg_1280.pt")
TTPLA_MD5 = "75c319e21af62bbf101188cf6d26dbc4"
TTPLA_SHA256 = "04026b2728af674fe4fca6f60999c61dfb2a0cecad3ff961e9e4ba9107e3a6f0"

# 可选 VGGT 深度缓存（28 帧 .npz，打包 zip），同为私有分发：原作者提供 zip 后
# 放到 data/vggt_depth_28frames.zip（或用 WPR_VGGT_URL 指向下载地址），
# --with-vggt-depth 会解压到 data/vggt_depth/。
VGGT_URL = os.environ.get("WPR_VGGT_URL", "")
VGGT_SHA256 = os.environ.get("WPR_VGGT_SHA256", "")   # 原作者分发时提供


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download_http(url, dest, md5=None, sha256=None):
    import requests
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {dest} already present", flush=True)
    else:
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
                    print(f"\r  {done/1e6:.1f}/{total/1e6:.1f} MB", end="",
                          flush=True)
        print(f"\n[done] {dest} ({done/1e6:.1f} MB)", flush=True)
    if md5:
        got = md5_of(dest)
        if got != md5:
            raise SystemExit(f"[FAIL] md5 mismatch for {dest}: {got} != {md5}")
        print(f"[ok]   md5 {md5}", flush=True)
    if sha256:
        got = sha256_of(dest)
        if got != sha256:
            raise SystemExit(f"[FAIL] sha256 mismatch for {dest}: {got} != {sha256}")
        print(f"[ok]   sha256 {sha256}", flush=True)


def download_ttpla():
    if not os.path.exists(TTPLA_DEST):
        if TTPLA_URL:
            try:
                download_http(TTPLA_URL, TTPLA_DEST, md5=TTPLA_MD5,
                              sha256=TTPLA_SHA256)
                return
            except Exception as e:
                print(f"\n[ttpla] 下载失败: {e}", flush=True)
        print("\n[ttpla] TTPLA checkpoint 为私有分发，未随官方源发布。", flush=True)
        print("[ttpla] 请向原作者索取该文件，手动放入:", flush=True)
        print(f"[ttpla]   {TTPLA_DEST}", flush=True)
        print(f"[ttpla] 放置后重新运行本脚本即可校验（md5={TTPLA_MD5}）",
              flush=True)
        return
    # 文件已就位：无论来自手动放置还是下载，都校验哈希。
    if md5_of(TTPLA_DEST) != TTPLA_MD5:
        raise SystemExit(f"[FAIL] md5 mismatch for {TTPLA_DEST}: "
                         f"expected {TTPLA_MD5}")
    if sha256_of(TTPLA_DEST) != TTPLA_SHA256:
        raise SystemExit(f"[FAIL] sha256 mismatch for {TTPLA_DEST}: "
                         f"expected {TTPLA_SHA256}")
    print(f"[ok]   {TTPLA_DEST} 校验通过 (md5 {TTPLA_MD5})", flush=True)


def download_vggt_depth():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    dest_dir = os.path.join(data_dir, "vggt_depth")
    if os.path.exists(dest_dir) and len(os.listdir(dest_dir)) >= 27:
        print(f"[skip] {dest_dir} already present ({len(os.listdir(dest_dir))} files)",
              flush=True)
        return
    import requests
    zip_path = os.path.join(data_dir, "vggt_depth_28frames.zip")
    if not os.path.exists(zip_path):
        if not VGGT_URL:
            print("\n[vggt] 深度缓存为私有分发。请向原作者索取 "
                  "vggt_depth_28frames.zip，", flush=True)
            print(f"[vggt] 放到 {zip_path} 后重新运行本脚本（或设置 WPR_VGGT_URL "
                  "指向下载地址）。", flush=True)
            print("[vggt] 缺失时管线仍可运行：人分支自动退化为最近簇兜底。",
                  flush=True)
            return
        print(f"[start] downloading {VGGT_URL} -> {zip_path}", flush=True)
        r = requests.get(VGGT_URL, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done/1e6:.1f}/{total/1e6:.1f} MB", end="",
                          flush=True)
        print(f"\n[done] {zip_path} ({done/1e6:.1f} MB)", flush=True)
    if VGGT_SHA256:
        got = sha256_of(zip_path)
        if got != VGGT_SHA256:
            raise SystemExit(f"[FAIL] sha256 mismatch for {zip_path}: "
                             f"{got} != {VGGT_SHA256}")
        print(f"[ok]   sha256 {VGGT_SHA256}", flush=True)
    # 只解压顶层 *.npz（防 zip-slip），落到 data/vggt_depth/<stem>.npz
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist()
                 if not n.startswith("/") and ".." not in n
                 and n.endswith(".npz") and "/" not in n]
        for n in names:
            target = os.path.join(dest_dir, n)
            if not os.path.exists(target):
                with z.open(n) as src, open(target, "wb") as dst:
                    dst.write(src.read())
    print(f"[done] {len(names)} npz -> {dest_dir}", flush=True)
    os.remove(zip_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-owlv2", action="store_true",
                    help="同时下载 OWLv2（可选兜底检测器，需 "
                         "requirements-owlv2.txt）")
    ap.add_argument("--with-vggt-depth", action="store_true",
                    help="同时下载 28 帧 VGGT 深度缓存到 data/vggt_depth/")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download, hf_hub_download

    # OWLv2 (transformers open-vocab detector, optional)
    if args.with_owlv2:
        owl_dest = os.path.join(WEIGHTS_DIR, "owlv2")
        if not (os.path.exists(os.path.join(owl_dest, "config.json"))
                and os.path.exists(os.path.join(owl_dest, "model.safetensors"))):
            print(f"[start] downloading {MODEL_ID_OWLV2} -> {owl_dest}", flush=True)
            snapshot_download(MODEL_ID_OWLV2, local_dir=owl_dest)
        else:
            print(f"[skip] {owl_dest} already present", flush=True)

    # YOLO-World (open-vocab detector, default)
    download_http(YOLO_URL, YOLO_DEST, md5=YOLO_MD5)

    # CLIP ViT-B/32 (text encoder YOLO-World's set_classes needs)
    download_http(CLIP_URL, CLIP_DEST, md5=CLIP_MD5)

    # SAM2 (mask refinement)
    sam2_dest = os.path.join(WEIGHTS_DIR, "sam2", SAM2_CKPT)
    if not os.path.exists(sam2_dest):
        print(f"[start] downloading {SAM2_REPO}/{SAM2_CKPT} -> {sam2_dest}",
              flush=True)
        hf_hub_download(SAM2_REPO, SAM2_CKPT,
                        local_dir=os.path.dirname(sam2_dest))
    got = md5_of(sam2_dest)
    if got != SAM2_MD5:
        raise SystemExit(f"[FAIL] md5 mismatch for {sam2_dest}: {got} != {SAM2_MD5}")
    print(f"[ok]   md5 {SAM2_MD5}", flush=True)

    # TTPLA (per_frame wire mode; distribution may be private, see weights/README.md)
    download_ttpla()

    if args.with_vggt_depth:
        download_vggt_depth()

    print("\n[all done] 校验通过的权重已就位；缺失的 TTPLA 条目按上方指引手动放置。",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
