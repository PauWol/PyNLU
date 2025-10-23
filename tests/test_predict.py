import json
import joblib
import numpy as np
from pathlib import Path

from pynlu import PyNLU

# Module-level DummyClf so joblib can pickle it
class DummyClf:
    def __init__(self, classes=None):
        if classes is None:
            classes = ["turn_on_lights", "noop"]
        self.classes_ = np.array(classes)

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        probs = np.zeros((n, len(self.classes_)))
        probs[:, 0] = 0.99
        if probs.shape[1] > 1:
            probs[:, 1:] = (1.0 - probs[:, :1]) / (probs.shape[1] - 1)
        return probs


def make_dummy_models_dir(tmp_path):
    assets = tmp_path / "assets"
    models_dir = assets / "models"
    models_dir.mkdir(parents=True)

    # Minimal meta.json so PyNLU.load_model can restore patterns and slot specs
    meta = {
        "languages": ["en"],
        "intent_slot_specs": {
            "turn_on_lights": {"ROOM": {"options": ["living room"], "type": "str"}},
            "noop": {}
        },
        "intent_patterns": {
            "en": {"turn_on_lights": [".*"], "noop": [".*"]}
        },
        "trained_meta": {}
    }
    (models_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # Dump a module-level DummyClf instance
    joblib.dump(DummyClf(classes=["turn_on_lights", "noop"]), models_dir / "en_nlpclf.joblib")
    return assets


def test_main_init(tmp_path, monkeypatch):
    assets = make_dummy_models_dir(tmp_path)

    # Prevent spaCy downloads: patch the util loader before PyNLU is created
    monkeypatch.setattr("pynlu.util.load_class_lang_models", lambda cls: None)

    pyn = PyNLU(str(assets / "models"))


    assert pyn  # type: ignore


def test_predict(tmp_path, monkeypatch):
    assets = make_dummy_models_dir(tmp_path)

    # IMPORTANT: patch the spaCy loader before creating PyNLU so no downloads occur
    monkeypatch.setattr("pynlu.util.load_class_lang_models", lambda cls: None)

    pyn = PyNLU(str(assets / "models"))

    text = "Turn on the light in the living room please"

    res = pyn.process(text)  # many implementations return either 3- or 4-tuple
    if isinstance(res, tuple) and len(res) == 3:
        intent, confidence, lang = res
        slots = pyn.slots(text, intent=intent, lang=lang)
    else:
        # assume (intent, confidence, lang, slots)
        intent, confidence, lang = res

    assert intent is not None
    assert isinstance(confidence, float)
    assert isinstance(lang, str)
    assert isinstance(slots, dict)
    room_val = slots.get("ROOM") or slots.get("room") or next(iter(slots.values()), "")
    assert "living" in str(room_val).lower()
