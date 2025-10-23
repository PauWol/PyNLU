import re
from langdetect import detect
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------- Load spaCy models ----------
nlp_models = {
    'en': spacy.load("en_core_web_sm"),
    'de': spacy.load("de_core_news_sm")
}

# ---------- Training data ----------
# English
train_texts_en = [
    "turn on the lights",
    "turn off all lights",
    "set temperature to 22",
    "play some music"
]
train_labels_en = ["turn_on_lights", "turn_off_lights", "set_temperature", "play_music"]

# German
train_texts_de = [
    "schalte das licht an",
    "schalte alle lichter aus",
    "stelle die temperatur auf 22",
    "spiele musik"
]
train_labels_de = ["turn_on_lights", "turn_off_lights", "set_temperature", "play_music"]

# ---------- Train classifiers ----------
vectorizer_en = TfidfVectorizer()
X_en = vectorizer_en.fit_transform(train_texts_en)
clf_en = LogisticRegression()
clf_en.fit(X_en, train_labels_en)

vectorizer_de = TfidfVectorizer()
X_de = vectorizer_de.fit_transform(train_texts_de)
clf_de = LogisticRegression()
clf_de.fit(X_de, train_labels_de)

# ---------- Prediction ----------
def predict_intent(text):
    lang = detect(text)  # Detect language
    if lang.startswith('de'):
        X_test = vectorizer_de.transform([text])
        intent = clf_de.predict(X_test)[0]
    else:
        X_test = vectorizer_en.transform([text])
        intent = clf_en.predict(X_test)[0]
    return intent, lang

# ---------- Slot extraction ----------
def extract_slots(text, lang):
    doc = nlp_models[lang[:2]](text)
    slots = {}

    # Extract numbers (temperature)
    for token in doc:
        if token.like_num:
            slots['number'] = int(token.text)

    # Extract locations (example)
    locations_en = ["kitchen", "living room", "bedroom"]
    locations_de = ["küche", "wohnzimmer", "schlafzimmer"]

    locations = locations_de if lang.startswith('de') else locations_en
    for loc in locations:
        if re.search(loc, text, re.IGNORECASE):
            slots['location'] = loc

    return slots

# ---------- Example ----------
user_input = "schalte das licht in der küche an"
intent, lang = predict_intent(user_input)
slots = extract_slots(user_input, lang)

print("Language:", lang)
print("Intent:", intent)
print("Slots:", slots)
