import copy
import json
import logging
from typing import List, Dict, Union

logger = logging.getLogger(__name__)


class Embedding:
    def __init__(self, name: str, source: str, keyword: str):
        self.name = name
        self.source = source
        self.keyword = keyword

    @classmethod
    def from_dict(cls, data: Dict) -> "Embedding":
        return cls(name=data["name"], source=data["source"], keyword=data["keyword"])

    def to_dict(self) -> Dict:
        return {"name": self.name, "source": self.source, "keyword": self.keyword}


class Lora:
    def __init__(
        self, name: str, src: str, trigger: str, weight: float, inject_when: List[str]
    ):
        self.name = name
        self.src = src
        self.trigger = trigger
        self.weight = weight
        self.inject_when = inject_when

    @classmethod
    def from_dict(cls, data: Dict) -> "Lora":
        return cls(
            name=data["name"],
            src=data["src"],
            trigger=data["trigger"],
            weight=data["weight"],
            inject_when=data["inject_when"],
        )

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "src": self.src,
            "trigger": self.trigger,
            "weight": self.weight,
            "inject_when": self.inject_when,
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
        examples: List[List[Union[str, int]]],
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
            "Model": self.model,
            "Path": self.path,
            "ModelType": self.model_type,
            "Parent": self.parent,
            "Description": self.description,
            "Generation": self.generation,
            "Aspect_Ratio": self.aspect_ratio,
            "Embeddings": {
                "positive": [e.to_dict() for e in self.embeddings["positive"]],
                "negative": [e.to_dict() for e in self.embeddings["negative"]],
            },
            "Loras": [lora.to_dict() for lora in self.loras],
            "Examples": self.examples,
        }
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def update(self, priority_values):
        """used to apply new values to an existing object (mostly used for merge)"""
        if hasattr(priority_values, "model") and priority_values.model:
            self.model = priority_values.model
        if hasattr(priority_values, "path") and priority_values.path:
            self.path = priority_values.path
        if hasattr(priority_values, "model_type") and priority_values.model_type:
            self.model_type = priority_values.model_type
        if hasattr(priority_values, "parent") and priority_values.parent:
            self.parent = priority_values.parent
        if hasattr(priority_values, "description") and priority_values.description:
            self.description = priority_values.description
        if hasattr(priority_values, "generation") and priority_values.generation:
            self.generation.update(priority_values.generation)

        # now lists and dicts
        if hasattr(priority_values, "aspect_ratio") and priority_values.aspect_ratio:
            self.aspect_ratio.update(priority_values.aspect_ratio)
        if hasattr(priority_values, "embeddings") and priority_values.embeddings:
            # FIXME: think about splitting in positive and negative, or full merge?
            if len(priority_values.embeddings["positive"]) > 0:
                self.embeddings["positive"] = priority_values.embeddings["positive"].copy()
            if len(priority_values.embeddings["negative"]) > 0:
                self.embeddings["negative"] = priority_values.embeddings["negative"].copy()

        if hasattr(priority_values, "loras") and priority_values.loras:
            self.loras.update(priority_values.loras)
        if hasattr(priority_values, "examples") and priority_values.examples:
            for element in priority_values.examples:
                if element not in self.examples:
                    self.examples.append(element)
            #self.examples.update(priority_values.examples)

    @classmethod
    def split_aspect_ratio(cls, aspect_ratio: str) -> tuple:
        try:
            # TODO: add unittest
            width = int(aspect_ratio.split("x")[0])
            height = int(aspect_ratio.split("x")[1])
            return width, height
        except Exception as e:
            logger.warning(f"Error splitting aspect ratios '{aspect_ratio}': {e}")
            return 512, 512

    @classmethod
    def from_dict(cls, item: dict) -> "ModelConfig":

        # Embeddings optional
        embeddings = {"positive": [], "negative": []}
        try:
            if "Embeddings" in item:
                embeddings["positive"] = [
                    Embedding.from_dict(e)
                    for e in item["Embeddings"].get("positive", [])
                ]
                embeddings["negative"] = [
                    Embedding.from_dict(e)
                    for e in item["Embeddings"].get("negative", [])
                ]
        except (KeyError, TypeError):
            pass

        # Loras optional
        loras = []
        try:
            if "Loras" in item:
                loras = [Lora.from_dict(lora_dict) for lora_dict in item.get("Loras", [])]
        except (KeyError, TypeError):
            pass

        if "Model" not in item:
            raise ValueError("Property 'Model' is missing")

        # create config with optional params
        config = cls(
            model=item["Model"],  # einziges Pflichtfeld
            path=item.get("Path", ""),
            model_type=item.get("ModelType", ""),
            parent=item.get("Parent", ""),
            description=item.get("Description", ""),
            generation=item.get("Generation", {}),
            aspect_ratio=item.get("Aspect_Ratio", {}),
            embeddings=embeddings,
            loras=loras,
            examples=item.get("Examples", []),
        )

        return config

    def sanity_check(self) -> bool:
        """
        Performs a sanity check on the ModelConfig instance to ensure all required fields are present
        and properly structured.

        Returns:
            bool: True if all checks pass, False otherwise

        The method checks:
        - Model name is present and not empty
        - Path is present and not empty
        - Aspect ratio dictionary exists and contains at least one entry
        - Generation dictionary exists
        - Embeddings dictionary exists and contains both 'positive' and 'negative' keys
        - Loras list exists
        - Examples list exists
        """
        try:
            # Check model name
            if not self.model or not isinstance(self.model, str):
                logger.warning(f"Model {self.model} has invalid model name")
                return False

            # Check model type
            if not self.model_type or not isinstance(self.model_type, str):
                logger.warning(f"Model {self.model} has invalid or no model_type")
                return False
            if self.model_type.lower() not in ["flux", "sdxl", "sd1.5"]:
                logger.warning(f"Model Type '{self.model_type}' is not allowed. it must be sd1.5, sdxl or flux.")

            # # check for common settings like infrence 4 on flux schnell and 50 on dev
            # TODO V2
            # if self.model_type == "flux" and "dev" in self.model:
            #     self.check_flux_dev()
            # elif self.model_type == "flux" and "schnell" in self.model:
            #     self.check_flux_schnell()
            # elif self.model_type == "sdxl":
            #     self.check_sdxl()

            # Check path
            if not self.path or not isinstance(self.path, str):
                logger.warning(f"Model {self.model} has invalid path")
                return False

            # Check aspect ratio
            if not isinstance(self.aspect_ratio, dict):
                logger.warning(f"Model {self.model} has invalid aspect_ratio type")
                return False

            if len(self.aspect_ratio) == 0:
                logger.warning(f"Model {self.model} has no aspect ratios defined")
                return False

            # Check generation dictionary
            if not isinstance(self.generation, dict):
                logger.warning(f"Model {self.model} has invalid generation settings type")
                return False

            # Check embeddings
            if not isinstance(self.embeddings, dict):
                logger.warning(f"Model {self.model} has invalid embeddings type")
                return False

            if 'positive' not in self.embeddings or 'negative' not in self.embeddings:
                logger.warning(f"Model {self.model} is missing positive or negative embeddings")
                return False

            if not isinstance(self.embeddings['positive'], list) or \
                    not isinstance(self.embeddings['negative'], list):
                logger.warning(f"Model {self.model} has invalid embeddings list type")
                return False

            # Check loras
            if not isinstance(self.loras, list):
                logger.warning(f"Model {self.model} has invalid loras type")
                return False

            # Check examples
            if not isinstance(self.examples, list):
                logger.warning(f"Model {self.model} has invalid examples type")
                return False

            return True

        except Exception as e:
            logger.warning(f"Sanity check failed for model {self.model}: {str(e)}")
            return False

    @classmethod
    def create_config_list_from_json(cls, json_str: str) -> List["ModelConfig"]:
        data = json.loads(json_str)
        configs = []

        for item in data:
            config = cls.from_dict(item)
            configs.append(config)

        return configs

    @classmethod
    def list_to_json(cls, configs: List["ModelConfig"]) -> str:
        # Nutze die existierende to_json-Methode und extrahiere die Daten
        data = [json.loads(config.to_json()) for config in configs]
        return json.dumps(data, indent=4)

    @classmethod
    def merge(
        cls, parentconfig: "ModelConfig", childconfig: "ModelConfig"
    ) -> "ModelConfig":
        if parentconfig is None:
            return childconfig
        if childconfig is None:
            return parentconfig
        logger.debug(f"Merge model {childconfig.model} with {parentconfig.model}")

        # deep copy to avoid overwriting parent if multiple childs refrencing to it
        result = cls.from_dict(parentconfig.to_dict())
        result.update(childconfig)
        return result

    @classmethod
    def get_config(cls, model: str, configs: List["ModelConfig"]) -> "ModelConfig":
        """get config include inherit values from parent"""
        logger.info(f"get inherited config for {model}")
        selectedmodelconfig = next(
            (config for config in configs if config.model == model), None
        )
        if selectedmodelconfig is None:
            logger.warning(f"Modelconfig does not contain configuration for {model}. Searching Default")
            selectedmodelconfig = next((config for config in configs if config.model == "default"), None)

        def find_parent(model: "ModelConfig", configs: List["ModelConfig"]):
            """recursive find and merge models"""
            if model is None:
                return
            merged = model
            if model.parent != "":
                parentconfig = next((config for config in configs if config.model == model.parent), None)
                if parentconfig is None:
                    logger.error(f"Model {model.parent} not found in modelconfigs")
                else:
                    mergedparent = find_parent(model=parentconfig, configs=configs)
                    merged = cls.merge(parentconfig=mergedparent, childconfig=model)

            return merged

        return find_parent(model=selectedmodelconfig, configs=configs)
