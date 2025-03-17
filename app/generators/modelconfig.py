import copy
import json, logging
from typing import List, Dict, Union

logger = logging.getLogger(__name__)

class Embedding:
    def __init__(self, name: str, source: str, keyword: str):
        self.name = name
        self.source = source
        self.keyword = keyword

    @classmethod
    def from_dict(cls, data: Dict) -> 'Embedding':
        return cls(
            name=data['name'],
            source=data['source'],
            keyword=data['keyword']
        )

    def to_dict(self) -> Dict:
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
    def from_dict(cls, data: Dict) -> 'Lora':
        return cls(
            name=data['name'],
            src=data['src'],
            trigger=data['trigger'],
            weight=data['weight'],
            inject_when=data['inject_when']
        )

    def to_dict(self) -> Dict:
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
        model_type: str,
        parent: str,
        description: str,
        generation: Dict,
        aspect_ratio: Dict,
        embeddings: Dict[str, List[Embedding]],
        loras: List[Lora],
        examples: List[List[Union[str, int]]]
    ):
        self.model = model
        self.path = path
        self.model_type = model_type
        self.parent = parent
        self.description = description
        self.generation = generation
        self.aspect_ratio = aspect_ratio
        self.embeddings = embeddings
        self.loras = loras
        self.examples = examples

    def to_dict(self) -> dict:
        data = {
            'Model': self.model,
            'Path': self.path,
            'Type': self.model_type,
            'Parent': self.parent,
            'Description': self.description,
            'Generation': self.generation,
            'Aspect_Ratio': self.aspect_ratio,
            'Embeddings': {
                'positive': [e.to_dict() for e in self.embeddings['positive']],
                'negative': [e.to_dict() for e in self.embeddings['negative']]
            },
            'Loras': [l.to_dict() for l in self.loras],
            'Examples': self.examples
        }
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)
    
    def update(self, new_values):
        if hasattr(new_values, 'model') and new_values.model:
            self.model = new_values.model
        if hasattr(new_values, 'path') and new_values.path:
            self.path = new_values.path
        if hasattr(new_values, 'model_type') and new_values.model_type:
            self.model_type = new_values.model_type
        if hasattr(new_values, 'parent') and new_values.parent:
            self.parent = new_values.parent
        if hasattr(new_values, 'description') and new_values.description:
            self.description = new_values.description
        if hasattr(new_values, 'generation') and new_values.generation:
            self.generation = new_values.generation

        # now lists and dicts
        if hasattr(new_values, 'aspect_ratio') and new_values.aspect_ratio:
            self.aspect_ratio = new_values.aspect_ratio.copy()
        if hasattr(new_values, 'embeddings') and new_values.embeddings:
            # FIXME: think about splitting in positive and negative, or full merge?
            if len(new_values.embeddings["positive"])>0:
                self.embeddings["positive"] = new_values.embeddings["positive"].copy()
            if len(new_values.embeddings["negative"])>0:
                self.embeddings["negative"] = new_values.embeddings["negative"].copy()
        
        if hasattr(new_values, 'loras') and new_values.loras:
            self.loras = new_values.loras.copy()
            # FIXME: todo append
        if hasattr(new_values, 'examples') and new_values.examples:
            if len(new_values.examples)>0:
                self.examples = new_values.examples.copy()

    @classmethod
    def from_dict(cls, item: dict) -> 'ModelConfig':

        # Embeddings optional
        embeddings = {'positive': [], 'negative': []}
        try:
            if 'Embeddings' in item:
                embeddings['positive'] = [Embedding.from_dict(e) for e in item['Embeddings'].get('positive', [])]
                embeddings['negative'] = [Embedding.from_dict(e) for e in item['Embeddings'].get('negative', [])]
        except (KeyError, TypeError):
            pass
            
        # Loras optional
        loras = []
        try:
            if 'Loras' in item:
                loras = [Lora.from_dict(l) for l in item.get('Loras', [])]
        except (KeyError, TypeError):
            pass
        
        if not "Model" in item:
            raise ValueError("Property 'Model' is missing")
        
        # create config with optional params
        config = cls(
            model=item['Model'],  # einziges Pflichtfeld
            path=item.get('Path', ''),
            model_type=item.get('ModelType', ''),
            parent=item.get('Parent', ''),
            description=item.get('Description', ''),
            generation=item.get('Generation', ''),
            aspect_ratio=item.get('Aspect_Ratio', ''),
            embeddings=embeddings,
            loras=loras,
            examples=item.get('Examples', [])
        )
            
        return config

    @classmethod
    def create_config_list_from_json(cls, json_str: str) -> List['ModelConfig']:
        data = json.loads(json_str)
        configs = []
        
        for item in data:
            config = cls.from_dict(item)
            configs.append(config)
            
        return configs

    @classmethod
    def list_to_json(cls, configs: List['ModelConfig']) -> str:
        # Nutze die existierende to_json-Methode und extrahiere die Daten
        data = [json.loads(config.to_json()) for config in configs]
        return json.dumps(data, indent=4)

    @classmethod
    def merge(cls, parentconfig: 'ModelConfig', childconfig: 'ModelConfig') -> 'ModelConfig':
        if parentconfig==None: return childconfig
        if childconfig==None: return parentconfig

        # deep copy
        result = cls.from_dict(parentconfig.to_dict())
        #result= copy.deepcopy(parentconfig)
        result.update(childconfig)
        return result

        p_json = parentconfig.to_json()
        c_json = childconfig.to_json()

        result = copy.deepcopy(p_json)
        result.update(c_json)
        m = cls.create_config_list_from_json(result)[0]
        return m

    @classmethod
    def get_config(cls, model: str, configs: List['ModelConfig']) -> 'ModelConfig':
        """get config include inherit values from parent"""
        selectedmodelconfig = next((config for config in configs if config.model == model), None)
        if selectedmodelconfig == None:
            logger.warning(f"Modelconfig does not contain configuration for {model}. Searching Default")

        selectedmodelconfig = next((config for config in configs if config.model == "default"), None)

        def find_parent(model, configs):
            c=next((config for config in configs if config.model == model), None)
            merged = c
            if c.parent != "":
                p = find_parent(c.parent, configs)
                if (p==None):
                    logger.error(f"Model Parent {model} not found in modelconfig.json")
                else:
                    merged=cls.merge(parent, child)
               
            return merged

        