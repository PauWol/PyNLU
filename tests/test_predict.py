from pynlu import PyNLU

def test_main_init():
    pyn = PyNLU("../assets/models")
    assert pyn
    
def test_predict():
    pyn = PyNLU("../assets/models")
    intent, confidence, lang, slots = pyn.predict("Turn on the light in the living room please")
    print(intent, confidence, lang, slots)
    assert intent
    assert confidence
    assert lang
    assert slots
    assert "living_room" in slots
    assert "turn_on_lights" in intent 