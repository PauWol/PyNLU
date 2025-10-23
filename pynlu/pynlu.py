# PyNLU
# Description: A simple Natural Language Understanding (NLU) library using spaCy.

from pathlib import Path
import warnings
import joblib, json, re

from . import util
from langdetect import detect

class PyNLU:
    def __init__(self, model_path_dir: str|None = None):
        self.model_path_dir = Path(model_path_dir) # type: ignore
        

        self.nlp_models = {}
        self.languages = []
        self.classifiers = {}
        self._meta = {}

        self._init()


    def _init(self):
        self._file_checks()
        self._load_data()
        self._languages()
        self._load_model()
        util.load_class_lang_models(self)
    
    def _file_checks(self):
        if not self.model_path_dir.exists():
            raise FileNotFoundError(f"Model directory {self.model_path_dir} does not exist.")
        if not self.model_path_dir.is_dir():
            raise ValueError(f"Model directory {self.model_path_dir} is not a directory.")
        if not any(self.model_path_dir.iterdir()):
            raise ValueError(f"Model directory {self.model_path_dir} is empty.")
        if not all(f.suffix.lower() in (".joblib", ".json") for f in self.model_path_dir.iterdir()):
            raise ValueError("Only .joblib model files are supported.")
        if not self.model_path_dir.joinpath("meta.json").exists():
            raise FileNotFoundError("meta.json not found in model directory.")
        
    
    def _load_data(self):
            self._meta = json.loads(self.model_path_dir.joinpath("meta.json").absolute().read_text(encoding="utf-8"))
    
    def _languages(self):
        self.languages =  self._meta.get("languages", []) # type: ignore
        if len(self.languages) == 0:
            raise ValueError("No languages specified.")
    
    def _load_model(self):
        for lang in self.languages:
            model_file = self.model_path_dir / f"{lang}_nlpclf.joblib"
            if not model_file.exists():
                warnings.warn(f"No classifier file for {lang} at {model_file}; skipping.")
                continue
            obj = joblib.load(model_file)   # pipeline or (clf, vec) tuple
            self.classifiers[lang] = obj

        # 3) Restore intent_slot_specs and compile regex patterns
        self.intent_slot_specs = self._meta.get("intent_slot_specs", {})
        self.intent_patterns = {}
        for lang, intents in self._meta.get("intent_patterns", {}).items():
            self.intent_patterns[lang] = {}
            for intent, pat_list in intents.items():
                compiled = []
                for pstr in pat_list:
                    try:
                        compiled.append(re.compile(pstr, re.IGNORECASE | re.UNICODE))
                    except re.error:
                        # fallback to escaped literal
                        compiled.append(re.compile(re.escape(pstr), re.IGNORECASE | re.UNICODE))
                self.intent_patterns[lang][intent] = compiled

        # 4) Save other meta if present
        self._trained_meta = self._meta.get("trained_meta", {})
        # done


    @staticmethod
    def language(text: str):
        """
        Detect the language of a given text.

        Parameters
        ----------
        text : str
            The input text to detect the language of.

        Returns
        -------
        str
            The detected language code (e.g. "en", "de", ...).
        """
        return detect(text)
    

    def predict(self, text: str):
        """
        Predict intent, confidence, language, and slots for a given text.

        Parameters
        ----------
        text : str
            The input text to predict.

        Returns
        -------
        intent : str
            The predicted intent.
        confidence : float
            The confidence of the prediction.
        lang : str
            The language of the input text.
        slots : dict[str, str]
            The predicted slots.
        """
        intent, confidence, lang = self.process(text)
        slots = self.slots(text, intent, lang)
        return intent, confidence, lang, slots
    
    def slots(self, text: str, intent: str|None = None, lang: str|None = None):
        """
        Predict slots for a given text and intent.

        Parameters
        ----------
        text : str
            The input text to predict slots for.
        intent : str, optional
            The intent to predict slots for. If not provided, will be predicted.
        lang : str, optional
            The language of the input text. If not provided, will be detected.

        Returns
        -------
        dict[str, str]
            The predicted slots, where keys are slot names and values are slot values.
        """
        if not intent or not lang:
            intent, _, lang = self.process(text)

        slots: dict[str, str] = {}
        patterns = self.intent_patterns.get(lang, {}).get(intent, [])
        slot_specs = self.intent_slot_specs.get(intent, {})

        # 2) try compiled patterns
        for pat in patterns:
            m = pat.search(text)
            if not m:
                continue
            gd = {k: v.strip() for k, v in (m.groupdict() or {}).items() if v}
            # type casting e.g. int
            for k, spec in slot_specs.items():
                if k in gd and isinstance(spec, dict) and spec.get("type") == "int":
                    digits = re.findall(r"-?\d+", str(gd[k]))
                    if digits:
                        gd[k] = int(digits[0])
            if gd:
                return gd

        # 3) options substring fallback
        for k, spec in slot_specs.items():
            if isinstance(spec, dict) and spec.get("options"):
                for opt in spec["options"]:
                    if re.search(fr"\b{re.escape(str(opt))}\b", text, re.IGNORECASE):
                        slots[k] = str(opt)

        # 4) spaCy NER fallback for free_text slots
        if any(isinstance(v, dict) and v.get("free_text") for v in slot_specs.values()):
            nlp = self.nlp_models.get(lang)
            if nlp:
                doc = nlp(text)
                ents_text = {ent.text for ent in doc.ents}
                for k, spec in slot_specs.items():
                    if isinstance(spec, dict) and spec.get("free_text") and k not in slots:
                        if ents_text:
                            slots[k] = next(iter(ents_text))

        return slots

    
    def process(self, text: str):
        """
        Process a text string and return the predicted intent, confidence, and language.

        Steps:

        1. Language selection: if only one language is supported, use that.
           Otherwise, detect the language using langdetect and use the appropriate classifier.
           If the detected language is not supported, fall back to the first supported language.

        2. Get the classifier object for the selected language.

        3. Predict the intent: if the classifier is a pipeline, use its predict_proba method.
           Otherwise, assume it is a tuple of (clf, vectorizer) and use the clf's predict_proba method.

        Returns:

        intent (str): the predicted intent
        confidence (float): the confidence of the prediction (0..1)
        lang (str): the language of the text
        """
        if len(self.languages) == 1:
            lang = self.languages[0]
        else:
            lang = util.lang_detect(text)  # your existing util.lang_detect uses langdetect.detect()
            if lang not in self.classifiers:
                # fallback to first supported language
                lang = next(iter(self.classifiers.keys()))

        # 2) get classifier object for lang
        clf_obj = self.classifiers.get(lang)
        if clf_obj is None:
            raise RuntimeError(f"No classifier loaded for language {lang}")

        # 3) predict — handle pipeline vs (clf, vectorizer) tuple
        try:
            # pipeline has predict_proba and accepts raw text list
            probs = clf_obj.predict_proba([text])[0]
            classes = clf_obj.classes_
        except Exception:
            # assume (clf, vectorizer)
            clf, vect = clf_obj
            X = vect.transform([text])
            probs = clf.predict_proba(X)[0]
            classes = clf.classes_

        top_idx = int(probs.argmax())
        intent = classes[top_idx]
        confidence = float(probs[top_idx])
        return intent, confidence, lang

    
if __name__ == "__main__":
    pynlu = PyNLU("./models")
    print("start")
    print(pynlu.nlp_models)