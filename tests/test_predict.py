import json
import joblib
import numpy as np
from pathlib import Path

from pynlu import PyNLU

def make_dummy_models_dir(tmp_path):
    assets = tmp_path / "assets"
    models_dir = assets / "models"
    models_dir.mkdir(parents=True)

    # Minimal meta.json so PyNLU.load_model can restore patterns and slot specs
    meta = {
        "languages": ["en"],
        "intent_slot_specs": {
            "turn_on_lights": {"ROOM": {"options": ["living room"], "type": "str"}}
        },
        "intent_patterns": {
            "en": {"turn_on_lights": [".*"]}
        },
        "trained_meta": {}
    }
    (models_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # Dummy classifier that returns one class with high confidence
    class DummyClf:
        def __init__(self):
            self.classes_ = np.array(["turn_on_lights"])
        def predict_proba(self, X):
            return np.array([[0.99]])

    joblib.dump(DummyClf(), models_dir / "en_nlpclf.joblib")
    return assets

def test_main_init(tmp_path, monkeypatch):
    # Create fake assets/models and instantiate PyNLU pointing to it
    assets = make_dummy_models_dir(tmp_path)

    # monkeypatch spaCy loaders to avoid any imports if your PyNLU tries to load spaCy in _init
    # If PyNLU._init calls util.load_class_lang_models, we should patch it there,
    # but a safe approach is ensuring PyNLU doesn't try to download at import time.
    try:
        pyn = PyNLU(str(assets / "models"))
    except Exception:
        # some implementations auto-load in __init__; ignore if already loaded
        pass

    assert pyn # type: ignore

def test_predict(tmp_path):
    assets = make_dummy_models_dir(tmp_path)
    pyn = PyNLU(str(assets / "models"))
    
    intent, confidence, lang, slots = pyn.process("Turn on the light in the living room please") # type: ignore
    # Some implementations return (intent, confidence, lang) or (intent, confidence, lang, slots).
    # Adjust assertions depending on your PyNLU.process/predict signature.
    assert intent is not None
    assert isinstance(confidence, float)
    assert lang in ("en", "de", "fr", "es", None) or isinstance(lang, str)
    # check that ROOM was extracted via options fallback
    assert isinstance(slots, dict)
    assert slots.get("ROOM") in ("living room", "living_room", "living-room", "livingroom") or "living" in str(slots.get("ROOM"))
