from pathlib import Path
import yaml

class Train:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self._data: dict|None = None
        
        self._file_checks()
        
    @property
    def languages(self):
        
        return self._data.get("languages", []) # type: ignore
    
    def _file_checks(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File {self.file_path} does not exist.")
        if self.file_path.suffix.lower() != ".yaml" and self.file_path.suffix.lower() != ".yml":
            raise ValueError("Only .yaml files are supported.")
        
        
    def _load_data(self):
        with open(self.file_path.absolute(), "r", encoding="utf-8") as file:
            self._data = yaml.safe_load(file)
    
    def _eval_config(self):
        pass
        
    def train(self):
        pass
    
    def save(self):
        pass
    
    
    def _classefier(self):
        for lang in self.languages:
            pass
    
    
    
if __name__ == "__main__":
    trainer = Train("../assets/config.yaml")
    trainer.train()
    trainer.save()