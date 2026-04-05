"""
ModelRegistry: pluggable model registry with auto-download support.

Models are tracked in models.json. On first use, weights are cached to ./models/.
"""
import json
import os
import torch
from pathlib import Path

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models.json")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class ModelRegistry:
    _registry: dict | None = None

    @classmethod
    def load(cls) -> dict:
        if cls._registry is None:
            with open(REGISTRY_PATH) as f:
                cls._registry = json.load(f)
        return cls._registry

    @classmethod
    def get_expert_id(cls, model_type: str, **kwargs) -> str:
        """Get canonical expert ID for a model type + params combo."""
        if model_type == "compressai":
            quality = kwargs.get("quality", 1)
            return f"img_mbt2018_q{quality}"
        elif model_type == "encodec":
            return "aud_encodec_24k"
        raise ValueError(f"Unknown model type: {model_type}")

    @classmethod
    def ensure_model(cls, expert_id: str) -> str:
        """
        Ensure model weights are cached locally. Download if not found.
        Returns local path to the weights.
        """
        registry = cls.load()
        if expert_id not in registry:
            raise ValueError(f"Unknown expert ID: {expert_id!r}. Known: {list(registry.keys())}")

        info = registry[expert_id]
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), info["local_path"])

        if os.path.exists(local_path):
            return local_path

        # Download the model
        print(f"[registry] This file requires the [{expert_id}] expert. Downloading weights...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if info["type"] == "compressai":
            from compressai.zoo import mbt2018_mean
            quality = info["quality"]
            print(f"[registry] Fetching mbt2018_mean quality={quality} from CompressAI...")
            model = mbt2018_mean(quality=quality, pretrained=True)
            torch.save(model.state_dict(), local_path)
            print(f"[registry] Saved → {local_path}")
        elif info["type"] == "encodec":
            from encodec import EncodecModel
            print(f"[registry] Fetching EnCodec 24kHz from Meta...")
            model = EncodecModel.encodec_model_24khz()
            torch.save(model.state_dict(), local_path)
            print(f"[registry] Saved → {local_path}")

        return local_path

    @classmethod
    def load_compressai_model(cls, expert_id: str):
        """Load a CompressAI model, using cached weights if available."""
        registry = cls.load()
        info = registry[expert_id]
        assert info["type"] == "compressai"

        local_path = cls.ensure_model(expert_id)
        from compressai.zoo import mbt2018_mean
        model = mbt2018_mean(quality=info["quality"], pretrained=False)
        model.load_state_dict(torch.load(local_path, map_location="cpu"))
        model.eval()
        return model

    @classmethod
    def list_models(cls) -> list[dict]:
        """List all registered models with their availability status."""
        registry = cls.load()
        result = []
        for expert_id, info in registry.items():
            local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), info["local_path"])
            result.append({
                "id": expert_id,
                "description": info.get("description", ""),
                "type": info["type"],
                "cached": os.path.exists(local_path),
                "local_path": local_path,
            })
        return result
