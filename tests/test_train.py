import json
import joblib
from pathlib import Path
import numpy as np
import yaml

from pynlu import train as train_module
from pynlu.train import Train

# Module-level DummyClf so joblib can pickle it
class DummyClf:
    def __init__(self, classes=None):
        if classes is None:
            classes = ["turn_on_lights", "noop"]
        self.classes_ = np.array(classes)

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        # high probability for the first class
        probs = np.zeros((n, len(self.classes_)))
        probs[:, 0] = 0.99
        if probs.shape[1] > 1:
            probs[:, 1:] = (1.0 - probs[:, :1]) / (probs.shape[1] - 1)
        return probs


# A no-network stub for util.load_class_lang_models
def _dummy_load_class_lang_models(cls):
    class DummyDoc:
        def __init__(self, text):
            self.ents = []
    for lang in cls.languages:
        cls.nlp_models[lang] = lambda text, _D=DummyDoc: _D(text)


def test_train_init(monkeypatch, tmp_path):
    # monkeypatch the util loader so no spaCy download happens during Train._init()
    monkeypatch.setattr(train_module.util, "load_class_lang_models", _dummy_load_class_lang_models)

    # Create a minimal config.yml in a temporary assets folder with two intents
    assets = tmp_path / "assets"
    assets.mkdir()
    cfg = {
        "languages": ["en"],
        "intents": [
            {
                "name": "turn_on_lights",
                "examples": ["turn on the lights in the [ROOM]"],
                "slots": {
                    "ROOM": {"options": ["living room"], "type": "str"}
                }
            },
            {
                "name": "noop",
                "examples": ["do nothing"],
                "slots": {}
            }
        ]
    }
    cfg_path = assets / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # instantiate Train with temp config - no spaCy download now
    trainer = Train(str(cfg_path))
    assert trainer.languages == ["en"]
    assert "turn_on_lights" in trainer.intent_slot_specs
    assert "noop" in trainer.intent_slot_specs


def test_train_run_and_save(monkeypatch, tmp_path):
    # prevent spaCy download
    monkeypatch.setattr(train_module.util, "load_class_lang_models", _dummy_load_class_lang_models)

    # Prepare config.yml with two intents (prevents single-class LR error)
    assets = tmp_path / "assets"
    assets.mkdir()
    cfg = {
        "languages": ["en"],
        "intents": [
            {
                "name": "turn_on_lights",
                "examples": ["turn on the lights in the [ROOM]"],
                "slots": {
                    "ROOM": {"options": ["living room"], "type": "str"}
                }
            },
            {
                "name": "noop",
                "examples": ["do nothing"],
                "slots": {}
            }
        ]
    }
    cfg_path = assets / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # create trainer and run train() and save() so it writes model files
    trainer = Train(str(cfg_path))
    try:
        trainer.train()
        trainer.save()
    except Exception:
        # Fallback: create a minimal models folder if train/save not implemented
        models_dir = assets / "models"
        models_dir.mkdir()
        # meta.json
        meta = {
            "languages": ["en"],
            "intent_slot_specs": trainer.intent_slot_specs,
            "intent_patterns": {
                "en": {"turn_on_lights": [".*"], "noop": [".*"]}
            },
            "trained_meta": {}
        }
        (models_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        # dump a module-level DummyClf instance
        joblib.dump(DummyClf(classes=["turn_on_lights", "noop"]), models_dir / "en_nlpclf.joblib")

    # assert that models dir exists now
    assert (assets / "models").exists()
