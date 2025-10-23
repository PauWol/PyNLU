from pynlu import PyNLU

pyn = PyNLU("./assets/models")
print("ww")
intent , confidence, lang = pyn.process("Wie wird das Wtter heute in Schweich")
print(intent, confidence, lang)

slots = pyn.slots("Wie wird das Wtter heute in Schweich", intent, lang)
print(slots)