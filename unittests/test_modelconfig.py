# Unittests
import unittest, json
from app.generators.modelconfig import ModelConfig


class TestModelConfig(unittest.TestCase):
    def setUp(self):
        self.sample_json = '''[
            {
                "Model": "TestModel",
                "Path": "test/path",
                "ModelType": "FLUX",
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
        configs = ModelConfig.create_config_list_from_json(self.sample_json)
        self.assertEqual(len(configs), 1)
        config = configs[0]
        self.assertEqual(config.model, "TestModel")
        self.assertEqual(config.path, "test/path")
        self.assertEqual(config.model_type, "FLUX")
        self.assertEqual(config.description, "Test description")
        self.assertEqual(config.generation["steps"], 40)
        self.assertEqual(len(config.embeddings["positive"]), 1)
        self.assertEqual(len(config.loras), 1)
        self.assertEqual(len(config.examples), 1)

    def test_to_json(self):
        configs = ModelConfig.create_config_list_from_json(self.sample_json)
        first_model = configs[0]
        json_str = first_model.to_json()
        data = json.loads(json_str)
        self.assertEqual(len(data), 10)
        self.assertEqual(data["Model"], first_model.model)
        self.assertEqual(data["Loras"][0]["name"], first_model.loras[0].name)

    def test_list_to_json(self):
        configs = ModelConfig.create_config_list_from_json(self.sample_json)
        # Zweite Config zum Testen der Liste erstellen
        config2 = ModelConfig(
            model="Model2",
            path="path2",
            model_type="SDXL",
            parent="",
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
        single_json = json.loads(configs[0].to_json())
        self.assertEqual(data[0], single_json)

    def test_merge(self):
        configs = ModelConfig.create_config_list_from_json(self.sample_json)
        parent = configs[0]

        child = ModelConfig.from_dict(
                {
                    "Model": "cm",
                    "Description": "--test--",
                    "Aspect_Ratio": {
                        "Square": "1024x1024"
                    }
                }
        )
        child.parent = parent.model

        merged = ModelConfig.merge(parentconfig=parent, childconfig=child)

        self.assertEqual(merged.model, child.model)
        self.assertEqual(merged.description, child.description)

        self.assertDictEqual(merged.aspect_ratio, child.aspect_ratio)

    def test_getconfig(self):
        configs = ModelConfig.create_config_list_from_json(self.sample_json)
        existing = configs[0]

        found = ModelConfig.get_config(existing.model, configs)
        self.assertIsNotNone(found, "the model should be found but is not")
        self.assertEqual(existing.model, found.model)
        self.assertEqual(existing.description, found.description)

class TestModelConfigMerge(unittest.TestCase):
    def setUp(self):
        # JSON-Konfigurationen für Parent und Child
        parent_json = '''
        [{
            "Model": "parent_model",
            "Path": "/parent/path",
            "ModelType": "parent_type",
            "Parent": "",
            "Description": "Parent config",
            "Generation": "v1",
            "Aspect_Ratio": {"16:9":"123x456"},
            "Embeddings": {"positive": ["p1"], "negative": ["n1"]},
            "Loras": ["lora1"],
            "Examples": ["example1"]
        }]
        '''
        child_json = '''
        [{
            "Model": "child_model",
            "Path": "/child/path",
            "ModelType": "child_type",
            "Parent": "parent_model",
            "Description": "Child config",
            "Generation": "v2",

            "Embeddings": {"positive": ["p2"], "negative": ["n2"]}
        }]
        '''
        
        self.parent_config = ModelConfig.create_config_list_from_json(parent_json)[0]
        self.child_config = ModelConfig.create_config_list_from_json(child_json)[0]

    def test_merge_parent_none(self):
        result = ModelConfig.merge(None, self.child_config)
        self.assertEqual(result, self.child_config)
        self.assertEqual(result.model, self.child_config.model)
        self.assertEqual(result.path, self.child_config.path)

    def test_merge_child_none(self):
        result = ModelConfig.merge(self.parent_config, None)
        self.assertEqual(result, self.parent_config)
        self.assertEqual(result.model, self.parent_config.model)
        self.assertEqual(result.path, self.parent_config.path)

    def test_merge_both_configs(self):
        result = ModelConfig.merge(self.parent_config, self.child_config)
        
        self.assertIsNot(result, self.parent_config)
        self.assertIsNot(result, self.child_config)
        
        self.assertEqual(result.model, self.child_config.model)
        self.assertEqual(result.path, self.child_config.path)
        self.assertEqual(result.model_type, self.child_config.model_type)
        self.assertEqual(result.parent, self.child_config.parent)
        self.assertEqual(result.description, self.child_config.description)
        self.assertEqual(result.generation, self.child_config.generation)
        self.assertDictEqual(result.aspect_ratio, self.parent_config.aspect_ratio)
        self.assertListEqual(result.embeddings["positive"], self.child_config.embeddings["positive"])
        self.assertListEqual(result.embeddings["negative"], self.parent_config.embeddings["negative"])
        self.assertListEqual(result.loras, self.parent_config.loras)
        self.assertListEqual(result.examples, self.parent_config.examples)

    def test_merge_is_deepcopy(self):
        """Make sure that the merge is not referenced to the input object"""
        result = ModelConfig.merge(self.parent_config, self.child_config)
        result.model = 'modified'
        result.embeddings['positive'].append('new')
        result.loras.append('new_lora')
        
        self.assertNotEqual(result.model, self.child_config.model)
        #self.assertListNotEqual(result.embeddings["positive"], self.child_config.embeddings['positive'])
        self.assertGreater(len(result.embeddings["positive"]), len(self.child_config.embeddings['positive']))

        #self.assertListNotEqual(result.loras, self.child_config.loras)
        self.assertGreater(len(result.loras), len(self.child_config.loras))
