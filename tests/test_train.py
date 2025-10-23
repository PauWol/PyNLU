from pynlu.train import Train

def test_train_init():
    trainer = Train("./assets/config.yml")
    assert trainer
    assert trainer._data
    
def test_train():
    trainer = Train("./assets/config.yml")
    trainer.train()
    trainer.save()