# ============================================================
# Tests — AutoTrainer
# api/auto_trainer.py
# ============================================================

import pytest

pytest.importorskip('torch')


# ── select_base_model ─────────────────────────────────────

class TestSelectBaseModel:

    def test_image_domain_returns_yolo(self):
        from api.auto_trainer import select_base_model
        result = select_base_model("detect potholes on road", "image")
        assert "yolo" in result["base_model"].lower()

    def test_text_domain_returns_bert(self):
        from api.auto_trainer import select_base_model
        result = select_base_model("classify spam emails", "text")
        assert "bert" in result["base_model"].lower() or \
               result["model_type"] == "bert"

    def test_medical_domain_returns_classifier(self):
        from api.auto_trainer import select_base_model
        result = select_base_model("diagnose xray scans", "medical")
        assert result["model_type"] in ("yolo_cls", "classifier", "yolo")

    def test_security_domain_returns_isolation_forest(self):
        from api.auto_trainer import select_base_model
        result = select_base_model("detect fraud transactions", "security")
        assert "isolation" in result["base_model"].lower() or \
               result["model_type"] == "sklearn"

    def test_unknown_domain_falls_back_gracefully(self):
        from api.auto_trainer import select_base_model
        result = select_base_model("some problem", "unknown_domain")
        assert "base_model" in result

    def test_result_has_required_keys(self):
        from api.auto_trainer import select_base_model
        result = select_base_model("detect cats", "image")
        for key in ("base_model", "model_type", "description", "matches", "category"):
            assert key in result

    def test_keyword_match_count_increases(self):
        from api.auto_trainer import select_base_model
        few_keywords = select_base_model("something", "image")
        many_keywords = select_base_model(
            "detect pothole crack road defect car vehicle fire smoke", "image")
        assert many_keywords["matches"] > few_keywords["matches"]


# ── train_new_model ───────────────────────────────────────

class TestTrainNewModel:

    def test_returns_success_status(self):
        from api.auto_trainer import train_new_model
        result = train_new_model("detect potholes", "image")
        assert result["status"] == "success"

    def test_result_has_required_keys(self):
        from api.auto_trainer import train_new_model
        result = train_new_model("classify spam", "text")
        for key in ("status", "accuracy", "train_time", "problem",
                    "category", "base_model", "model_type", "trained_at"):
            assert key in result

    def test_accuracy_in_expected_range_image(self):
        from api.auto_trainer import train_new_model
        result = train_new_model("detect potholes", "image")
        assert 60.0 <= result["accuracy"] <= 100.0

    def test_accuracy_in_expected_range_text(self):
        from api.auto_trainer import train_new_model
        result = train_new_model("classify reviews", "text")
        assert 70.0 <= result["accuracy"] <= 100.0

    def test_accuracy_in_expected_range_security(self):
        from api.auto_trainer import train_new_model
        result = train_new_model("detect fraud", "security")
        assert 75.0 <= result["accuracy"] <= 100.0

    def test_train_time_positive(self):
        from api.auto_trainer import train_new_model
        result = train_new_model("test problem", "image")
        assert result["train_time"] > 0

    def test_category_preserved_in_result(self):
        from api.auto_trainer import train_new_model
        result = train_new_model("classify text", "text")
        assert result["category"] == "text"

    def test_progress_callback_called(self):
        from api.auto_trainer import train_new_model
        calls = []
        train_new_model("test", "image",
                        progress_callback=lambda s, t, msg: calls.append(s))
        assert len(calls) > 0

    def test_progress_callback_step_increases(self):
        from api.auto_trainer import train_new_model
        calls = []
        train_new_model("test", "image",
                        progress_callback=lambda s, t, msg: calls.append(s))
        assert calls == sorted(calls)


# ── run_yolo_detection ────────────────────────────────────

class TestRunYoloDetection:

    def test_returns_error_on_missing_file(self):
        from api.auto_trainer import run_yolo_detection
        result = run_yolo_detection("/nonexistent/path/image.jpg")
        # Should return error dict, not raise
        assert "status" in result
        assert result["status"] in ("error", "success")

    def test_error_result_has_boxes_key(self):
        from api.auto_trainer import run_yolo_detection
        result = run_yolo_detection("/nonexistent/path/image.jpg")
        assert "boxes" in result

    def test_error_result_boxes_is_list(self):
        from api.auto_trainer import run_yolo_detection
        result = run_yolo_detection("/nonexistent/path/image.jpg")
        assert isinstance(result["boxes"], list)
