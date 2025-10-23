from pynlu import PyNLU

pyn = PyNLU("./assets/models")

intent, confidence, lang, slots = pyn.predict("Turn on the light in the living room please")

print(intent, confidence, lang, slots)