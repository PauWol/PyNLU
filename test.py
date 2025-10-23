from pynlu.train import Train

if __name__ == "__main__":
    trainer = Train("./assets/config.yml")
    """     x = "Wie wird das Wtter heute in Schweich"
    intent, confidence, lang = trainer.process(x)
    print("Intent:", intent, f"confidence={confidence:.2%}", lang)
    slots = trainer.slots(x, intent, lang)
    print("Slots:", slots) """
    trainer.train()
    trainer.save()