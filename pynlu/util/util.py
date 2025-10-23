import subprocess
import sys

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from langdetect import detect, detect_langs
import re
from typing import Dict, List, Tuple, Pattern, Any


def is_model_installed(lang: str):
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


def download_model(model_name):
    """
    Download a spaCy model.

    Downloads the specified spaCy model if it is not installed.

    :param model_name: str
        The name of the spaCy model to download.

    :raises RuntimeError:
        If the model could not be downloaded.

    """
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download model {model_name}.\nError: {e}")


def load_lang_model(nlp_models:dict,lang: str, model_name: str):
    """
    Load a language model.

    Load a language model and store it in the provided dictionary.

    :param nlp_models: dict
        Dictionary containing the NLP models.
    :param lang: str
        Language code of the model to load.
    :param model_name: str
        Name of the spaCy model to load.

    """
    if not is_model_installed(lang):
        print(f"Model {model_name} is not installed. Downloading...")
        download_model(model_name)
    nlp_models[lang] = spacy.load(model_name)


def load_class_lang_models(cls):
    """
    Load language models for a class.

    This method loads language models for the specified class. The target class
    must have the following attributes to work properly:

    - nlp_models: dict - Dictionary containing the NLP models
    - languages: list[str] - List of supported language codes

    :param cls:
    :return:
    """
    for lang in cls.languages:
        match lang:
            case "en":
                load_lang_model(cls.nlp_models,"en", "en_core_web_sm")
            case "de":
                load_lang_model(cls.nlp_models,"de", "de_core_news_sm")
            case "fr":
                load_lang_model(cls.nlp_models,"fr", "fr_core_news_sm")
            case "es":
                load_lang_model(cls.nlp_models,"es", "es_core_news_sm")
            case _:
                raise ValueError(f"Language {lang} is not supported.")


def load_lang_classifier(lang: str, train_texts: list[str], train_labels: list[str]):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(train_texts)
    clf = LogisticRegression()
    clf.fit(X, train_labels)
    return clf, vectorizer

def load_class_lang_classifiers(cls):
    for lang in cls.languages:
        cls.classifiers[lang] = load_lang_classifier(lang, cls.train_texts[lang], cls.train_labels[lang])

def lang_detect(text: str):
        return detect(text)


def clean_example_text_for_langdetect(text: str) -> str:
    """
    Remove slot markers like [SLOT] / [SLOT:*] and collapse to neutral token to avoid
    skewing language detection.
    Keep literal optional words {..} since they are part of the language.
    """
    # Remove [SLOT] or [SLOT:*] markers and replace with a neutral token
    text = re.sub(r"\[[A-Za-z_][A-Za-z0-9_]*(?::\*)?\]", " X ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_example_language(example: str, allowed_langs: List[str]) -> str:
    """
    Detect language of a training example robustly:
    - Support explicit prefix 'xx:: ' to force language.
    - Strip placeholders before detection.
    - Use detect_langs and pick best among allowed languages.
    Fallback to detect() if needed; if result not allowed, return the first allowed.
    """
    # Explicit prefix e.g., 'de:: schalte ...'
    m = re.match(r"^([a-z]{2})::\s*(.+)$", example.strip(), re.IGNORECASE)
    if m:
        pref, rest = m.group(1).lower(), m.group(2)
        if pref in allowed_langs:
            return pref
        # If prefix not allowed, continue to detect on rest
        example = rest

    cleaned = clean_example_text_for_langdetect(example)
    try:
        langs = detect_langs(cleaned)
        # Choose the best candidate among allowed languages
        best = None
        best_prob = -1.0
        for cand in langs:
            code = cand.lang[:2]
            if code in allowed_langs and cand.prob > best_prob:
                best, best_prob = code, cand.prob
        if best:
            return best
        # Fallback: simple detect
        code = detect(cleaned)[:2]
        return code if code in allowed_langs else allowed_langs[0]
    except Exception:
        # On any failure, pick the first allowed language
        return allowed_langs[0]


# ---------------- Template-based slot extraction utilities ---------------- #

_SPACE_RE = re.compile(r"\s+")

def _escape_literal(text: str) -> str:
    """Escape literal text and normalize spaces to '\s+' for flexible matching."""
    text = text.strip()
    if not text:
        return ""
    esc = re.escape(text)
    # Replace escaped spaces with flexible whitespace
    esc = esc.replace(r"\ ", r"\s+")
    return esc


def _build_option_group(options: List[str]) -> str:
    escaped = [ _escape_literal(o) for o in options ]
    return "(?:" + "|".join(escaped) + ")"


def compile_example_to_regex(example: str, slot_specs: Dict[str, Dict[str, Any]]) -> Tuple[Pattern[str], List[str]]:
    """
    Compile an example string with placeholders into a regex pattern.

    Supported constructs:
    - [SLOT]       -> required slot, default span 1-4 tokens unless constrained
    - [SLOT:*]     -> free-text slot (multi-token)
    - {optional}   -> optional literal words/phrases

    Slot behavior is further controlled by slot_specs:
    - options: [..]  -> alternation group
    - free_text: true -> multi-token greedy but bounded by following literal
    - type: str/int   -> can be used later for casting

    Returns compiled pattern and list of slot names in order of appearance.
    """

    # Tokenize by finding slot and optional markers
    # Patterns: [NAME] or [NAME:*] and {optional text}
    token_re = re.compile(r"(\[[A-Za-z_][A-Za-z0-9_]*(?::\*)?\]|\{[^}]*\})")
    parts = token_re.split(example)

    slot_order: List[str] = []
    regex_parts: List[str] = [r"(?i)\b"]  # case-insensitive, start at word boundary

    # Lookahead helper to get following literal for bounding free_text
    def _next_literal(idx: int) -> str:
        for j in range(idx + 1, len(parts)):
            p = parts[j]
            if not p:
                continue
            if p.startswith("[") or p.startswith("{"):
                # skip markers, continue to look for literal
                continue
            literal = _escape_literal(p)
            if literal:
                return literal
        return ""

    def _next_is_slot(idx: int) -> bool:
        for j in range(idx + 1, len(parts)):
            p = parts[j]
            if not p:
                continue
            if p.startswith("{"):
                # optional literal, keep scanning
                continue
            return p.startswith("[")
        return False

    for i, p in enumerate(parts):
        if not p:
            continue
        if p.startswith("["):
            # Slot
            name_raw = p[1:-1]  # strip [ ]
            name, free_over = (name_raw.split(":", 1) + [""])[:2] if ":" in name_raw else (name_raw, "")
            name = name.strip()
            slot_order.append(name)

            spec = slot_specs.get(name, {})
            options = spec.get("options")
            free_text = spec.get("free_text", False) or (free_over.strip() == "*")

            if options:
                group = _build_option_group([str(o) for o in options])
                regex_parts.append(fr"(?P<{name}>{group})")
            elif free_text:
                following = _next_literal(i)
                if following:
                    # Bounded by the next literal using a lookahead only
                    regex_parts.append(fr"(?P<{name}>.+?)(?=\s+{following})")
                else:
                    # If the next meaningful token is another slot, bound to a small number of tokens
                    if _next_is_slot(i):
                        regex_parts.append(fr"(?P<{name}>\S+(?:\s+\S+){{0,2}})")
                    else:
                        # Capture until end if truly at the end
                        regex_parts.append(fr"(?P<{name}>.+)$")
            else:
                # Default: capture up to 1-4 tokens
                regex_parts.append(fr"(?P<{name}>\S+(?:\s+\S+){{0,3}})")
        elif p.startswith("{") and p.endswith("}"):
            # Optional literal
            literal = p[1:-1]
            esc = _escape_literal(literal)
            if esc:
                regex_parts.append(fr"(?:\s+{esc})?")
        else:
            # Literal text
            esc = _escape_literal(p)
            if esc:
                if regex_parts and not regex_parts[-1].endswith("\b"):
                    regex_parts.append(r"\s+")
                regex_parts.append(esc)

    # Allow trailing whitespace and end boundary
    regex_parts.append(r"\b")

    # Join and compile
    pattern_str = "".join(regex_parts)
    try:
        pattern = re.compile(pattern_str, re.IGNORECASE | re.UNICODE)
    except re.error:
        # Fallback: escape the whole example if something went wrong
        pattern = re.compile(_escape_literal(example), re.IGNORECASE | re.UNICODE)
        slot_order = []
    return pattern, slot_order