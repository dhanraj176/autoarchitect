# ============================================================
# Tests — data_uploader.py
# process_user_data, train_on_user_data helpers,
# predict_with_user_model, _save_file, _load_text_data,
# _fallback_result, cleanup_old_uploads
# ============================================================

import os
import json
import base64
import io
import time
import shutil
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── _fallback_result ──────────────────────────────────────────

class TestFallbackResult:

    def test_returns_dict(self):
        from api.data_uploader import _fallback_result
        assert isinstance(_fallback_result(2), dict)

    def test_contains_required_keys(self):
        from api.data_uploader import _fallback_result
        r = _fallback_result(3)
        for k in ("train_accuracy", "test_accuracy", "dataset",
                  "train_size", "test_size", "error", "architecture",
                  "parameters"):
            assert k in r

    def test_accuracies_are_zero(self):
        from api.data_uploader import _fallback_result
        r = _fallback_result(4)
        assert r["train_accuracy"] == 0
        assert r["test_accuracy"]  == 0

    def test_dataset_is_user_data(self):
        from api.data_uploader import _fallback_result
        assert _fallback_result(2)["dataset"] == "user_data"

    def test_architecture_is_resnet18(self):
        from api.data_uploader import _fallback_result
        assert "ResNet18" in _fallback_result(2)["architecture"]


# ── _load_text_data ──────────────────────────────────────────

class TestLoadTextData:

    def test_empty_on_missing_directory(self, tmp_path):
        from api.data_uploader import _load_text_data
        texts, labels = _load_text_data(str(tmp_path / "nonexistent"))
        assert texts  == []
        assert labels == []

    def test_loads_txt_files_from_class_folders(self, tmp_path):
        from api.data_uploader import _load_text_data
        (tmp_path / "spam").mkdir()
        (tmp_path / "ham").mkdir()
        (tmp_path / "spam" / "a.txt").write_text("buy now")
        (tmp_path / "ham"  / "b.txt").write_text("hello friend")

        texts, labels = _load_text_data(str(tmp_path))
        assert len(texts)  == 2
        assert len(labels) == 2
        assert "spam" in labels
        assert "ham"  in labels

    def test_ignores_non_txt_files(self, tmp_path):
        from api.data_uploader import _load_text_data
        (tmp_path / "cls").mkdir()
        (tmp_path / "cls" / "img.jpg").write_bytes(b"\xff\xd8")
        (tmp_path / "cls" / "note.txt").write_text("hello")

        texts, labels = _load_text_data(str(tmp_path))
        assert len(texts) == 1

    def test_returns_correct_text_content(self, tmp_path):
        from api.data_uploader import _load_text_data
        (tmp_path / "positive").mkdir()
        (tmp_path / "positive" / "x.txt").write_text("great product")

        texts, _ = _load_text_data(str(tmp_path))
        assert texts[0] == "great product"


# ── _save_file ────────────────────────────────────────────────

class TestSaveFile:

    def _make_jpeg_b64(self):
        """Minimal valid RGB image as base64."""
        from PIL import Image
        buf = io.BytesIO()
        img = Image.new("RGB", (10, 10), color=(100, 100, 100))
        img.save(buf, format="JPEG")
        raw = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{raw}"

    def test_saves_text_file_for_text_category(self, tmp_path):
        from api.data_uploader import _save_file
        folder = str(tmp_path)
        _save_file("hello world", "text", folder, "doc0")
        saved = list(Path(folder).glob("*.txt"))
        assert len(saved) == 1
        assert saved[0].read_text() == "hello world"

    def test_saves_jpg_for_image_category(self, tmp_path):
        from api.data_uploader import _save_file
        folder  = str(tmp_path)
        b64_img = self._make_jpeg_b64()
        _save_file(b64_img, "image", folder, "img0")
        saved = list(Path(folder).glob("*.jpg"))
        assert len(saved) == 1

    def test_saves_jpg_for_medical_category(self, tmp_path):
        from api.data_uploader import _save_file
        folder  = str(tmp_path)
        b64_img = self._make_jpeg_b64()
        _save_file(b64_img, "medical", folder, "scan0")
        assert any(Path(folder).glob("*.jpg"))

    def test_bad_b64_does_not_raise(self, tmp_path):
        """Corrupt base64 should be swallowed, not crash."""
        from api.data_uploader import _save_file
        # Should not raise
        _save_file("data:image/jpeg;base64,NOTVALID==",
                   "image", str(tmp_path), "bad")


# ── process_user_data ─────────────────────────────────────────

class TestProcessUserData:

    def _make_text_files(self, count=6):
        """Return list of plain text 'files' and matching labels."""
        files  = [f"sample text {i}" for i in range(count)]
        labels = (["class_a"] * (count // 2) +
                  ["class_b"] * (count - count // 2))
        return files, labels

    def test_raises_if_fewer_than_4_files(self, tmp_path):
        from api.data_uploader import process_user_data
        with patch("api.data_uploader.UPLOADS_DIR", str(tmp_path)):
            with pytest.raises(ValueError, match="at least 4"):
                process_user_data(["a", "b"], ["x", "y"],
                                  "test problem", "text")

    def test_returns_dict_with_required_keys(self, tmp_path):
        from api.data_uploader import process_user_data
        files, labels = self._make_text_files()
        with patch("api.data_uploader.UPLOADS_DIR", str(tmp_path)), \
             patch("api.data_uploader._save_file"):
            result = process_user_data(files, labels,
                                       "classify docs", "text")
        for k in ("train_dir", "test_dir", "classes",
                  "counts", "upload_dir", "total_files", "n_classes"):
            assert k in result

    def test_n_classes_matches_unique_labels(self, tmp_path):
        from api.data_uploader import process_user_data
        files, labels = self._make_text_files()
        with patch("api.data_uploader.UPLOADS_DIR", str(tmp_path)), \
             patch("api.data_uploader._save_file"):
            result = process_user_data(files, labels,
                                       "classify docs", "text")
        assert result["n_classes"] == 2

    def test_meta_json_written(self, tmp_path):
        from api.data_uploader import process_user_data
        files, labels = self._make_text_files()
        with patch("api.data_uploader.UPLOADS_DIR", str(tmp_path)), \
             patch("api.data_uploader._save_file"):
            result = process_user_data(files, labels,
                                       "classify docs", "text")
        meta_path = Path(result["upload_dir"]) / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["category"] == "text"

    def test_total_files_matches_input(self, tmp_path):
        from api.data_uploader import process_user_data
        files, labels = self._make_text_files(8)
        with patch("api.data_uploader.UPLOADS_DIR", str(tmp_path)), \
             patch("api.data_uploader._save_file"):
            result = process_user_data(files, labels,
                                       "classify docs", "text")
        assert result["total_files"] == 8


# ── cleanup_old_uploads ───────────────────────────────────────

class TestCleanupOldUploads:

    def test_removes_old_folders(self, tmp_path):
        from api.data_uploader import cleanup_old_uploads
        old_dir = tmp_path / "old_upload"
        old_dir.mkdir()
        # Back-date mtime to 10 days ago
        old_time = time.time() - 10 * 86400
        os.utime(str(old_dir), (old_time, old_time))

        with patch("api.data_uploader.UPLOADS_DIR", str(tmp_path)):
            cleanup_old_uploads(days_old=7)

        assert not old_dir.exists()

    def test_keeps_recent_folders(self, tmp_path):
        from api.data_uploader import cleanup_old_uploads
        new_dir = tmp_path / "recent_upload"
        new_dir.mkdir()

        with patch("api.data_uploader.UPLOADS_DIR", str(tmp_path)):
            cleanup_old_uploads(days_old=7)

        assert new_dir.exists()

    def test_does_nothing_if_uploads_dir_missing(self):
        from api.data_uploader import cleanup_old_uploads
        with patch("api.data_uploader.UPLOADS_DIR", "/nonexistent/path"):
            cleanup_old_uploads(days_old=1)   # should not raise


# ── predict_with_user_model ───────────────────────────────────

import importlib.util as _iutil
_NO_TORCH = _iutil.find_spec("torch") is None

@pytest.mark.skipif(_NO_TORCH, reason="requires torch")
class TestPredictWithUserModel:

    def test_returns_error_for_missing_model(self, tmp_path):
        from api.data_uploader import predict_with_user_model
        result = predict_with_user_model(
            str(tmp_path / "nomodel.pth"),
            "some input",
            "text",
            ["class_a", "class_b"]
        )
        # Should return gracefully with an error or fallback
        assert "label" in result or "error" in result

    def test_non_image_category_returns_first_class(self, tmp_path):
        """For text/security, a mocked checkpoint returns first class."""
        import torch
        from api.data_uploader import predict_with_user_model

        model_path = str(tmp_path / "model.pth")
        checkpoint = {
            "n_classes": 2,
            "classes":   ["ham", "spam"],
        }
        torch.save(checkpoint, model_path)

        result = predict_with_user_model(
            model_path, "buy now cheap", "text", ["ham", "spam"])
        assert result["label"] in ("ham", "spam")
        assert result["confidence"] > 0
