import warnings
from pathlib import Path
from pynlu import util
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import json
from datetime import datetime
import yaml
import re

class Train:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self._data: dict|None = None

        self.nlp_models = {}
        self.languages = []
        self.classifiers = {}

        self.train_texts = {
            "en": [],
            "de": [],
            "fr": [],
            "es": []
        }
        self.train_labels = {
            "en": [],
            "de": [],
            "fr": [],
            "es": []
        }

        # Intent -> slot specs (from config)
        self.intent_slot_specs: dict[str, dict] = {}
        # lang -> intent -> list[compiled regex Pattern]
        self.intent_patterns: dict[str, dict[str, list]] = {}

        self._init()


    def _init(self):
        self._file_checks()
        self._load_data()
        self._languages()
        self._eval_config()
        util.load_class_lang_models(self)
        util.load_class_lang_classifiers(self)

    def _languages(self):
        self.languages =  self._data.get("languages", []) # type: ignore
        if len(self.languages) == 0:
            raise ValueError("No languages specified.")
    
    def _file_checks(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File {self.file_path} does not exist.")
        if self.file_path.suffix.lower() != ".yaml" and self.file_path.suffix.lower() != ".yml":
            raise ValueError("Only .yaml files are supported.")
        
        
    def _load_data(self):
        with open(self.file_path.absolute(), "r", encoding="utf-8") as file:
            self._data = yaml.safe_load(file)
    
    def _eval_config(self):
        for intent in self._data.get("intents", []): # type: ignore
            label = intent.get("name")

            # Collect slot specs for this intent
            slots_cfg = intent.get("slots", {}) or {}
            self.intent_slot_specs[label] = slots_cfg

            # Prepare patterns container per lang
            for lang in self.languages:
                self.intent_patterns.setdefault(lang, {}).setdefault(label, [])

            # Build training data by language and compile patterns
            for example in intent.get("examples", []):
                # Robust language detection for examples with placeholders
                lang = util.detect_example_language(example, self.languages)
                if not lang in self.languages:
                    warnings.warn(f"Language '{lang}' of example '{example}' is not supported.\nCould be a false identification.Skipping example.")
                    continue
                # Clean placeholders out of training text so brackets/slots don't skew the classifier
                cleaned = util.clean_example_text_for_langdetect(example)
                self.train_texts[lang].append(cleaned)
                self.train_labels[lang].append(label)


                # Compile a regex pattern for slot extraction from this example
                pattern, _ = util.compile_example_to_regex(example, slots_cfg)
                self.intent_patterns.setdefault(lang, {}).setdefault(label, []).append(pattern)



        
    def train(self):
        for lang in self.languages:
            X = self.train_texts[lang]
            y = self.train_labels[lang]
            if len(X) < 2:
                warnings.warn(f"Not enough examples for {lang}; skipping.")
                continue
            if len(set(y)) < 2:
                warnings.warn(f"Only one class present for {lang}; classifier may be unusable.")
            pipe = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000, random_state=42))
            pipe.fit(X, y)
            self.classifiers[lang] = pipe
            # store metadata if you want:
            self._trained_meta = getattr(self, "_trained_meta", {})
            self._trained_meta[lang] = {"n_examples": len(X), "classes": list(set(y))}
    
    def save(self):
        model_dir = Path(self.file_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        for lang, obj in self.classifiers.items():
            fname = model_dir / f"{lang}_nlpclf.joblib"
            joblib.dump(obj, fname, compress=3)  # pipeline or (clf, vec) tuple

        # Save metadata + patterns + slot specs
        meta = {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "languages": self.languages,
            "intent_slot_specs": self.intent_slot_specs,
            # intent_patterns are compiled regex objects — convert to strings
            "intent_patterns": {
                lang: {intent: [p.pattern for p in pats] for intent, pats in iset.items()}
                for lang, iset in self.intent_patterns.items()
            },
            "trained_meta": getattr(self, "_trained_meta", {}),
        }
        with open(model_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def process(self,text: str) -> tuple[str, float, str]:
        if len(self.languages) != 1:
            lang = util.lang_detect(text)
        else:
            lang = self.languages[0]
        clf, vectorizer = self.classifiers[lang]
        X_test = vectorizer.transform([text])
        # Use probability estimates for confidence
        probs = clf.predict_proba(X_test)[0]
        classes = clf.classes_
        top_idx = probs.argmax()
        intent = classes[top_idx]
        confidence = float(probs[top_idx])  # 0..1
        return intent, confidence, lang

    def slots(self,text: str,intent: str|None = None,lang: str|None = None) -> dict[str, str]:
        
        # Use intent and lang if provided else detect
        if not intent or not lang:
            # Predict intent to know which patterns to try
            intent, intent_conf, lang = self.process(text)

        slots: dict[str, str] = {}
        patterns = self.intent_patterns.get(lang, {}).get(intent, []) # type: ignore
        slot_specs = self.intent_slot_specs.get(intent, {}) # type: ignore

        # Try to match any compiled pattern
        for pat in patterns:
            m = pat.search(text)
            if not m:
                continue
            gd = {k: v.strip() for k, v in m.groupdict().items() if v}
            # Basic post-processing: cast ints
            for k, spec in slot_specs.items():
                if k in gd and isinstance(spec, dict) and spec.get("type") == "int":
                    digits = re.findall(r"-?\d+", gd[k])
                    if digits:
                        gd[k] = int(digits[0])
            if gd:
                return gd

        # Fallbacks
        # 1) Options substring match
        for k, spec in slot_specs.items():
            if isinstance(spec, dict) and spec.get("options"):
                for opt in spec["options"]:
                    if re.search(fr"\b{re.escape(str(opt))}\b", text, re.IGNORECASE):
                        slots[k] = str(opt)

        # 2) spaCy NER fallback for free_text slots
        if any(isinstance(v, dict) and v.get("free_text") for v in slot_specs.values()):
            doc = self.nlp_models[lang](text)
            ents_text = {ent.text for ent in doc.ents}
            for k, spec in slot_specs.items():
                if isinstance(spec, dict) and spec.get("free_text") and k not in slots:
                    # Pick first entity mention as a heuristic
                    if ents_text:
                        slots[k] = next(iter(ents_text))

        return slots
    
if __name__ == "__main__":
    trainer = Train("../assets/config.yaml")
    x = "Hallo bitte gib mir das heutige Wetter"
    intent, confidence, lang = trainer.process(x)
    print("Intent:", intent, confidence, lang)