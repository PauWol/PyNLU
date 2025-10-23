# PyNLU
# Description: A simple Natural Language Understanding (NLU) library using spaCy.

from pathlib import Path
import subprocess
import sys

import spacy
from langdetect import detect

class PyNLU:
    def __init__(self, model_path: str|None = None,languages: list[str]|None = None):
        self.model_path = Path(model_path) if model_path is not None else None
        
        if languages is None or len(languages) == 0:
            print("No languages specified. Defaulting to English ('en').")
            self.languages = ["en"]
        else:
            self.languages = languages


        self.nlp_models = {}
    def _init(self):
        self._load_lang()
        
    def load_model(self, model_path: str|None = None):
        if model_path is not None:
            self.model_path = Path(model_path)
        if self.model_path is None:
            raise ValueError("Model path is not set.")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file {self.model_path} does not exist.")
        if self.model_path.suffix.lower() != ".ftz":
            raise ValueError("Only .ftz model files are supported.")
        
    @staticmethod
    def _is_model_installed(lang: str):
        models = spacy.util.get_installed_models()
        match lang:
            case "en":
                return "en_core_web_sm" in models
            case "de":
                return "de_core_news_sm" in models
            case "fr":
                return "fr_core_news_sm" in models
            case "es":
                return "es_core_news_sm" in models
            case _:
                return False
    
    @staticmethod        
    def download_model(model_name):
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download model {model_name}.\nError: {e}")

    def _lang_model(self,lang: str, model_name: str):
        if not self._is_model_installed(lang):
            print(f"Model {model_name} is not installed. Downloading...")
            self.download_model(model_name)
        self.nlp_models[lang] = spacy.load(model_name)

    def _load_lang(self):
        for lang in self.languages:
            match lang:
                case "en":
                    self._lang_model("en","en_core_web_sm")
                case "de":
                    self._lang_model("de","de_core_news_sm")
                case "fr":
                    self._lang_model("fr","fr_core_news_sm")
                case "es":
                    self._lang_model("es","es_core_news_sm")
                case _:
                    raise ValueError(f"Language {lang} is not supported.")
        
    @staticmethod
    def language(text: str):
        return detect(text)
    
    @staticmethod
    def intent(text: str):
        pass
    
    @staticmethod
    def slots(text: str):
        pass
    
    def process(self, text: str):
        pass
    
if __name__ == "__main__":
    pynlu = PyNLU(languages=["en","de"])
    pynlu._init()
    print(pynlu.nlp_models)