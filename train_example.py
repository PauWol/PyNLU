from pynlu.train import Train

if __name__ == "__main__":
    trainer = Train("./assets/config.yml")
    trainer.train()
    trainer.save()