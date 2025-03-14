import json
from typing import List, Dict, Union

class Embedding:
    def __init__(self, name: str, source: str, keyword: str):
        self.name = name
        self.source = source
        self.keyword = keyword

    @classmethod
    def from_json(cls, data: Dict) -> 'Embedding':
        return cls(
            name=data['name'],
            source=data['source'],
            keyword=data['keyword']
        )

    def to_json(self) -> Dict:
        return {
            'name': self.name,
            'source': self.source,
            'keyword': self.keyword
        }

class Lora:
    def __init__(self, name: str, src: str, trigger: str, weight: float, inject_when: List[str]):
        self.name = name
        self.src = src
        self.trigger = trigger
        self.weight = weight
        self.inject_when = inject_when

    @classmethod
    def from_json(cls, data: Dict) -> 'Lora':
        return cls(
            name=data['name'],
            src=data['src'],
            trigger=data['trigger'],
            weight=data['weight'],
            inject_when=data['inject_when']
        )

    def to_json(self) -> Dict:
        return {
            'name': self.name,
            'src': self.src,
            'trigger': self.trigger,
            'weight': self.weight,
            'inject_when': self.inject_when
        }

class ModelConfig:
    """
    Usage example:
    # JSON to Objekt
    configs = ModelConfig.from_json(json_str)
    
    # Erstes Config-Objekt aus der Liste
    config = configs[0]
    
    # Object to JSON
    json_output = config.to_json()
    print(json_output)
    """
    def __init__(
        self,
        model: str,
        path: str,
        type_: str,
        description: str,
        generation: Dict,
        aspect_ratio: Dict,
        embeddings: Dict[str, List[Embedding]],
        loras: List[Lora],
        examples: List[List[Union[str, int]]]
    ):
        self.model = model
        self.path = path
        self.type = type_
        self.description = description
        self.generation = generation
        self.aspect_ratio = aspect_ratio
        self.embeddings = embeddings
        self.loras = loras
        self.examples = examples

    @classmethod
    def from_json(cls, json_str: str) -> List['ModelConfig']:
        data = json.loads(json_str)
        configs = []
        
        for item in data:
            embeddings = {
                'positive': [Embedding.from_json(e) for e in item['Embeddings']['positive']],
                'negative': [Embedding.from_json(e) for e in item['Embeddings']['negative']]
            }
            
            loras = [Lora.from_json(l) for l in item['Loras']]
            
            config = cls(
                model=item['Model'],
                path=item['Path'],
                type_=item['Type'],
                description=item['Description'],
                generation=item['Generation'],
                aspect_ratio=item['Aspect_Ratio'],
                embeddings=embeddings,
                loras=loras,
                examples=item['Examples']
            )
            configs.append(config)
            
        return configs

    def to_json(self) -> str:
        data = {
            'Model': self.model,
            'Path': self.path,
            'Type': self.type,
            'Description': self.description,
            'Generation': self.generation,
            'Aspect_Ratio': self.aspect_ratio,
            'Embeddings': {
                'positive': [e.to_json() for e in self.embeddings['positive']],
                'negative': [e.to_json() for e in self.embeddings['negative']]
            },
            'Loras': [l.to_json() for l in self.loras],
            'Examples': self.examples
        }
        return json.dumps([data], indent=4)

    @classmethod
    def list_to_json(cls, configs: List['ModelConfig']) -> str:
        # Nutze die existierende to_json-Methode und extrahiere die Daten
        data = [json.loads(config.to_json())[0] for config in configs]
        return json.dumps(data, indent=4)

