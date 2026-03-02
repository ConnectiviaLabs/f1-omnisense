"""Dataset manager — CRUD, frame/audio ingestion, YOLO/HuggingFace export.

Ported from MediaSense dataset_controller.py + audio_dataset.py.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from omnidapt._config import get_config
from omnidapt._storage import get_storage
from omnidapt._types import DatasetInfo, DatasetStats, ModelDomain

logger = logging.getLogger(__name__)

COLLECTION = "datasets"
FRAMES_COLLECTION = "dataset_frames"
AUDIO_COLLECTION = "dataset_audio"


class DatasetManager:
    """Dataset CRUD + sample management + export."""

    def __init__(self):
        self._storage = get_storage()
        self._config = get_config()

    # ── CRUD ─────────────────────────────────────────────────────────────

    def create_dataset(
        self,
        name: str,
        domain: str | ModelDomain,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> DatasetInfo:
        """Create a new dataset."""
        if isinstance(domain, str):
            domain = ModelDomain(domain)

        existing = self._storage.find_one(COLLECTION, {"name": name})
        if existing:
            return DatasetInfo.from_dict(existing)

        ds = DatasetInfo(
            name=name,
            domain=domain,
            description=description,
            labels=labels or [],
        )
        self._storage.save_document(COLLECTION, ds.to_dict())
        logger.info("Created dataset '%s' (%s)", name, domain.value)
        return ds

    def get_dataset(self, name: str) -> Optional[DatasetInfo]:
        doc = self._storage.find_one(COLLECTION, {"name": name})
        if doc:
            return DatasetInfo.from_dict(doc)
        return None

    def list_datasets(self, domain: Optional[ModelDomain] = None) -> List[DatasetInfo]:
        query = {"domain": domain.value} if domain else None
        docs = self._storage.find_documents(COLLECTION, query)
        return [DatasetInfo.from_dict(d) for d in docs]

    def delete_dataset(self, name: str) -> bool:
        """Delete dataset and all its samples."""
        self._storage.delete_document(COLLECTION, {"name": name})
        # Delete all associated frames/audio
        for doc in self._storage.find_documents(FRAMES_COLLECTION, {"dataset_name": name}):
            self._storage.delete_document(FRAMES_COLLECTION, {"_id": doc["_id"]})
        for doc in self._storage.find_documents(AUDIO_COLLECTION, {"dataset_name": name}):
            self._storage.delete_document(AUDIO_COLLECTION, {"_id": doc["_id"]})

        # Clean up files
        ds_dir = self._config.datasets_dir / name
        if ds_dir.exists():
            shutil.rmtree(ds_dir)

        logger.info("Deleted dataset '%s'", name)
        return True

    # ── Frame Samples (CV) ───────────────────────────────────────────────

    def add_frames(
        self,
        dataset_name: str,
        frames: List[Dict[str, Any]],
    ) -> int:
        """Add image frames to a CV dataset.

        Each frame: {"image_data": "base64...", "label": "drone", "bbox": [x,y,w,h], ...}
        Ported from MediaSense dataset_controller.py save_frames().
        """
        ds = self.get_dataset(dataset_name)
        if not ds:
            ds = self.create_dataset(dataset_name, ModelDomain.CV)

        frames_dir = self._config.datasets_dir / dataset_name / "images"
        frames_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for i, frame in enumerate(frames):
            image_data = frame.get("image_data", "")
            if not image_data:
                continue

            # Strip data URL prefix
            if image_data.startswith("data:"):
                image_data = image_data.split(",", 1)[-1]

            try:
                img_bytes = base64.b64decode(image_data)
            except Exception:
                logger.debug("Invalid base64 in frame %d", i)
                continue

            # Save image file
            ext = frame.get("format", "jpg")
            filename = f"{int(time.time() * 1000)}_{saved}.{ext}"
            img_path = frames_dir / filename
            img_path.write_bytes(img_bytes)

            # Save metadata
            doc = {
                "dataset_name": dataset_name,
                "filename": filename,
                "path": str(img_path),
                "label": frame.get("label", ""),
                "bbox": frame.get("bbox"),
                "width": frame.get("width", 0),
                "height": frame.get("height", 0),
                "confidence": frame.get("confidence", 0),
                "created_at": time.time(),
            }
            # Include any extra metadata
            for k in ("source", "video_id", "time", "detection"):
                if k in frame:
                    doc[k] = frame[k]

            self._storage.save_document(FRAMES_COLLECTION, doc)
            saved += 1

        # Update dataset sample count
        count = self._storage.count_documents(FRAMES_COLLECTION, {"dataset_name": dataset_name})
        self._storage.update_document(
            COLLECTION, {"name": dataset_name},
            {"sample_count": count, "updated_at": time.time()},
        )

        # Update labels
        labels = set(ds.labels)
        for f in frames:
            if f.get("label"):
                labels.add(f["label"])
        if labels != set(ds.labels):
            self._storage.update_document(
                COLLECTION, {"name": dataset_name}, {"labels": sorted(labels)},
            )

        logger.info("Added %d frames to '%s'", saved, dataset_name)
        return saved

    # ── Audio Samples ────────────────────────────────────────────────────

    def add_audio_chunks(
        self,
        dataset_name: str,
        chunks: List[Dict[str, Any]],
    ) -> int:
        """Add audio chunks to an AUDIO dataset.

        Each chunk: {"audio_data": "base64...", "label": "drone_buzz", "duration": 2.5, ...}
        Ported from MediaSense audio_dataset.py.
        """
        ds = self.get_dataset(dataset_name)
        if not ds:
            ds = self.create_dataset(dataset_name, ModelDomain.AUDIO)

        audio_dir = self._config.datasets_dir / dataset_name / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for i, chunk in enumerate(chunks):
            audio_data = chunk.get("audio_data", "")
            if not audio_data:
                continue

            if audio_data.startswith("data:"):
                audio_data = audio_data.split(",", 1)[-1]

            try:
                audio_bytes = base64.b64decode(audio_data)
            except Exception:
                logger.debug("Invalid base64 in audio chunk %d", i)
                continue

            ext = chunk.get("format", "wav")
            filename = f"{int(time.time() * 1000)}_{saved}.{ext}"
            audio_path = audio_dir / filename
            audio_path.write_bytes(audio_bytes)

            doc = {
                "dataset_name": dataset_name,
                "filename": filename,
                "path": str(audio_path),
                "label": chunk.get("label", ""),
                "duration": chunk.get("duration", 0),
                "sample_rate": chunk.get("sample_rate", 16000),
                "created_at": time.time(),
            }
            for k in ("source", "segment_id", "metadata"):
                if k in chunk:
                    doc[k] = chunk[k]

            self._storage.save_document(AUDIO_COLLECTION, doc)
            saved += 1

        count = self._storage.count_documents(AUDIO_COLLECTION, {"dataset_name": dataset_name})
        self._storage.update_document(
            COLLECTION, {"name": dataset_name},
            {"sample_count": count, "updated_at": time.time()},
        )

        labels = set(ds.labels)
        for c in chunks:
            if c.get("label"):
                labels.add(c["label"])
        if labels != set(ds.labels):
            self._storage.update_document(
                COLLECTION, {"name": dataset_name}, {"labels": sorted(labels)},
            )

        logger.info("Added %d audio chunks to '%s'", saved, dataset_name)
        return saved

    # ── Export ────────────────────────────────────────────────────────────

    def export_yolo(self, dataset_name: str, train_split: float = 0.8) -> Path:
        """Export dataset in YOLO format (images + labels + data.yaml).

        Directory structure:
            export_dir/
            ├── data.yaml
            ├── train/
            │   ├── images/
            │   └── labels/
            └── val/
                ├── images/
                └── labels/
        """
        ds = self.get_dataset(dataset_name)
        if not ds:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        frames = self._storage.find_documents(FRAMES_COLLECTION, {"dataset_name": dataset_name})
        if not frames:
            raise ValueError(f"No frames in dataset '{dataset_name}'")

        export_dir = self._config.yolo_export_dir / dataset_name
        if export_dir.exists():
            shutil.rmtree(export_dir)

        for split in ("train", "val"):
            (export_dir / split / "images").mkdir(parents=True)
            (export_dir / split / "labels").mkdir(parents=True)

        # Build label map
        labels = sorted(set(f.get("label", "") for f in frames if f.get("label")))
        label_map = {l: i for i, l in enumerate(labels)}

        # Shuffle and split
        random.shuffle(frames)
        split_idx = int(len(frames) * train_split)
        splits = {"train": frames[:split_idx], "val": frames[split_idx:]}

        for split_name, split_frames in splits.items():
            for frame in split_frames:
                src = Path(frame.get("path", ""))
                if not src.exists():
                    continue

                dst = export_dir / split_name / "images" / src.name
                shutil.copy2(src, dst)

                # Write YOLO label
                label = frame.get("label", "")
                bbox = frame.get("bbox")
                if label in label_map and bbox:
                    label_file = export_dir / split_name / "labels" / f"{src.stem}.txt"
                    cls_id = label_map[label]
                    w = frame.get("width", 1)
                    h = frame.get("height", 1)
                    if w > 0 and h > 0:
                        # Convert [x, y, w, h] to YOLO format [cx, cy, nw, nh]
                        bx, by, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
                        cx = (bx + bw / 2) / w
                        cy = (by + bh / 2) / h
                        nw = bw / w
                        nh = bh / h
                        label_file.write_text(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        # Write data.yaml
        yaml_content = (
            f"path: {export_dir}\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"nc: {len(labels)}\n"
            f"names: {labels}\n"
        )
        (export_dir / "data.yaml").write_text(yaml_content)

        logger.info("Exported YOLO dataset to %s (%d frames, %d classes)", export_dir, len(frames), len(labels))
        return export_dir

    def export_huggingface_audio(self, dataset_name: str) -> Path:
        """Export audio dataset for HuggingFace (WAV files + metadata.json)."""
        ds = self.get_dataset(dataset_name)
        if not ds:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        chunks = self._storage.find_documents(AUDIO_COLLECTION, {"dataset_name": dataset_name})
        if not chunks:
            raise ValueError(f"No audio in dataset '{dataset_name}'")

        export_dir = self._config.audio_export_dir / dataset_name
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True)

        metadata = []
        for chunk in chunks:
            src = Path(chunk.get("path", ""))
            if not src.exists():
                continue

            dst = export_dir / src.name
            shutil.copy2(src, dst)
            metadata.append({
                "file_name": src.name,
                "label": chunk.get("label", ""),
                "duration": chunk.get("duration", 0),
                "sample_rate": chunk.get("sample_rate", 16000),
            })

        (export_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info("Exported HuggingFace audio to %s (%d chunks)", export_dir, len(chunks))
        return export_dir

    def export_zip(self, dataset_name: str) -> Path:
        """Export dataset as a ZIP file."""
        ds = self.get_dataset(dataset_name)
        if not ds:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        if ds.domain == ModelDomain.CV:
            export_dir = self.export_yolo(dataset_name)
        else:
            export_dir = self.export_huggingface_audio(dataset_name)

        zip_path = self._config.exports_dir / f"{dataset_name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(export_dir):
                for f in files:
                    fp = Path(root) / f
                    zf.write(fp, fp.relative_to(export_dir))

        return zip_path

    # ── Stats ────────────────────────────────────────────────────────────

    def get_stats(self, dataset_name: str) -> DatasetStats:
        ds = self.get_dataset(dataset_name)
        if not ds:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        if ds.domain == ModelDomain.CV:
            samples = self._storage.find_documents(FRAMES_COLLECTION, {"dataset_name": dataset_name})
        else:
            samples = self._storage.find_documents(AUDIO_COLLECTION, {"dataset_name": dataset_name})

        label_dist: Dict[str, int] = {}
        total_size = 0
        for s in samples:
            lbl = s.get("label", "unlabeled")
            label_dist[lbl] = label_dist.get(lbl, 0) + 1
            p = Path(s.get("path", ""))
            if p.exists():
                total_size += p.stat().st_size

        return DatasetStats(
            name=dataset_name,
            domain=ds.domain,
            total_samples=len(samples),
            label_distribution=label_dist,
            total_size_bytes=total_size,
            avg_sample_size_bytes=total_size // max(len(samples), 1),
        )

    def get_samples(
        self, dataset_name: str, limit: int = 50, offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get sample metadata (without binary data)."""
        ds = self.get_dataset(dataset_name)
        if not ds:
            return []

        collection = FRAMES_COLLECTION if ds.domain == ModelDomain.CV else AUDIO_COLLECTION
        all_docs = self._storage.find_documents(collection, {"dataset_name": dataset_name})
        return all_docs[offset : offset + limit]

    def validate_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Validate dataset quality."""
        ds = self.get_dataset(dataset_name)
        if not ds:
            return {"valid": False, "error": "Dataset not found"}

        stats = self.get_stats(dataset_name)
        issues = []

        if stats.total_samples == 0:
            issues.append("No samples")
        elif stats.total_samples < 10:
            issues.append(f"Very few samples ({stats.total_samples})")

        if len(stats.label_distribution) < 2:
            issues.append("Less than 2 classes")

        # Check class imbalance
        if stats.label_distribution:
            counts = list(stats.label_distribution.values())
            if max(counts) > 10 * min(counts):
                issues.append("Severe class imbalance")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "stats": stats.to_dict(),
        }


# ── Singleton ────────────────────────────────────────────────────────────

_manager: Optional[DatasetManager] = None


def get_dataset_manager() -> DatasetManager:
    global _manager
    if _manager is None:
        _manager = DatasetManager()
    return _manager


def reset_dataset_manager() -> None:
    global _manager
    _manager = None
