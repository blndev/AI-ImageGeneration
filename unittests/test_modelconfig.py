# Unittests
import unittest, json
from app.generators.modelconfig import ModelConfig


class TestModelConfig(unittest.TestCase):
    def setUp(self):
        self.sample_json = '''[
            {
                "Model": "TestModel",
                "Path": "test/path",
                "Type": "FLUX",
                "Description": "Test description",
                "Generation": {
                    "steps": 40,
                    "guidance": 4.0,
                    "negative_prompt": "False"
                },
                "Aspect_Ratio": {
                    "Square": "1024x1024",
                    "Landscape": "1152x768",
                    "Portrait": "768x1152"
                },
                "Embeddings": {
                    "positive": [
                        {
                            "name": "PosEmb",
                            "source": "pos.pt",
                            "keyword": "positive"
                        }
                    ],
                    "negative": [
                        {
                            "name": "NegEmb",
                            "source": "neg.pt",
                            "keyword": "negative"
                        }
                    ]
                },
                "Loras": [
                    {
                        "name": "TestLora",
                        "src": "lora.safetensors",
                        "trigger": "test",
                        "weight": 1.0,
                        "inject_when": ["keyword1", "keyword2"]
                    }
                ],
                "Examples": [
                    ["Test example", "style", "■ Landscape", 1]
                ]
            }
        ]'''

    def test_from_json(self):
        configs = ModelConfig.from_json(self.sample_json)
        self.assertEqual(len(configs), 1)
        config = configs[0]
        self.assertEqual(config.model, "TestModel")
        self.assertEqual(config.path, "test/path")
        self.assertEqual(config.type, "FLUX")
        self.assertEqual(config.description, "Test description")
        self.assertEqual(config.generation["steps"], 40)
        self.assertEqual(len(config.embeddings["positive"]), 1)
        self.assertEqual(len(config.loras), 1)
        self.assertEqual(len(config.examples), 1)

    def test_to_json(self):
        configs = ModelConfig.from_json(self.sample_json)
        json_str = configs[0].to_json()
        data = json.loads(json_str)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["Model"], "TestModel")
        self.assertEqual(data[0]["Loras"][0]["name"], "TestLora")

    def test_list_to_json(self):
        configs = ModelConfig.from_json(self.sample_json)
        # Zweite Config zum Testen der Liste erstellen
        config2 = ModelConfig(
            model="Model2",
            path="path2",
            type_="SDXL",
            description="desc2",
            generation={"steps": 30, "guidance": 3.0, "negative_prompt": "True"},
            aspect_ratio={"Square": "1024x1024"},
            embeddings={"positive": [], "negative": []},
            loras=[],
            examples=[["example2", "style2", "■ Square", 2]]
        )
        
        json_str = ModelConfig.list_to_json([configs[0], config2])
        data = json.loads(json_str)
        
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["Model"], "TestModel")
        self.assertEqual(data[1]["Model"], "Model2")
        self.assertEqual(data[0]["Type"], "FLUX")
        self.assertEqual(data[1]["Type"], "SDXL")
        # Zusätzlicher Test um Konsistenz mit to_json zu verifizieren
        single_json = json.loads(configs[0].to_json())[0]
        self.assertEqual(data[0], single_json)