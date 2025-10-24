# PyNLU - Python Natural Language Understanding Library

PyNLU is a lightweight Python NLU library for **intent recognition** and **slot extraction** using `spaCy` and `scikit-learn`. It supports multiple languages (English, German, French, Spanish) and is designed for easy deployment and integration in applications.

---

## Features

* Multi-language support: English (`en`), German (`de`), French (`fr`), Spanish (`es`).
* Intent classification using TF-IDF + Logistic Regression.
* Slot extraction with template-based regex and spaCy NER fallback.
* Config-driven setup via `.yml` files.
* Easy training, saving, and loading of models.
* Lightweight, modular, and easy to extend.

---

## Installation

### Option 1: Install directly via pip from GitHub (recommended for simple usage)

This installs PyNLU as a regular package without needing to clone the repo or modify source files:

```bash
pip install git+https://github.com/PauWol/PyNLU.git
```

*This option is recommended for most users who only want to use PyNLU without editing the source.*


### Option 2: Install from GitHub source (for development / source manipulation)

- Step-1 Clone the repository
  ```bash
  git clone https://github.com/PauWol/PyNLU.git
  cd PyNLU
  ```
- Step-2 `Optional`: create a virtual environment (recommended)
  ```bash
  python -m venv venv
  source venv/bin/activate  # Linux/macOS
  venv\Scripts\activate     # Windows
  ```
- Step-3 Install dependencies
  - Either using `requirements.txt`
    ```bash
    pip install -r requirements.txt
    ```
  - Or `setup.py` as actual package
     ```bash
    python setup.py install
    ```



---

## Project Structure

```
PyNLU/
├── assets/
│   ├── config.yml        # Training configuration
│   └── models/           # Trained models (auto-generated)
├── pynlu/                # Source code
├── examples/             # Example scripts and usage
└── README.md
```

---

## Config File (`config.yml`)

The config file defines **languages**, **intents**, **examples**, and **slots**.

Example:

```yaml
languages: [en, de]

intents:
  - name: get_weather_now
    examples:
      - "get the weather in [LOCATION]"
      - "Wie ist das Wetter in [LOCATION]"
    slots:
      LOCATION:
        type: str
        free_text: true
```

* `languages`: List of supported language codes.
* `intents`: Each intent has a `name`, `examples` for training, and `slots` with type/options.
* `slots`: Can have `type`, `free_text`, and `options` for predefined values.

---

## Training

Train models from a `.yml` config:

```python
from pynlu.train import Train

trainer = Train("./assets/config.yml")
trainer.train()
trainer.save()
```

* `train()`: Trains intent classifiers per language.
* `save()`: Saves models to `assets/models` for later use.

---

## Using PyNLU

Load trained models and perform predictions:

```python
from pynlu import PyNLU

pyn = PyNLU("./assets/models")

intent, confidence, lang, slots = pyn.predict("Wie wird das Wetter heute in London")
print(intent, confidence, lang, slots)
```

* `predict(text)`: Returns a tuple `(intent, confidence, language, slots)`.
* Slot extraction handles placeholders, free-text slots, and option matching.

---

## Example Intents

* **Weather**: `get_weather_now`, `get_weather_tomorrow`
* **Lights Control**: `turn_on_lights`, `turn_off_lights`
* **Music**: `play_music`, `stop_music`
* **Smart Home**: `set_temperature`, `set_alarm`, `control_door`
* **Others**: `check_time`, `check_news`, `open_curtains`, `close_curtains`, `control_tv`, `add_to_shopping_list`

Slots can be configured for free text or predefined options.

---

## Usage Tips

* Keep your `.yml` configuration clean and consistent.
* Include diverse examples per intent to improve classifier accuracy.
* Update spaCy models for each language you use.
* After modifying config, always retrain models.

---

## License

MIT License

---

## Contributing

Feel free to contribute by adding new intents, languages, or improving slot extraction logic. Open an issue or submit a pull request on GitHub.

---

## References

* [spaCy](https://spacy.io/)
* [scikit-learn](https://scikit-learn.org/)
* [langdetect](https://pypi.org/project/langdetect/)
* YAML configuration for NLU templates
