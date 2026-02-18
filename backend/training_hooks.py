"""
Training hooks for capturing embeddings during ACE-Step LoRA training.
Captures from three spaces:
1. Audio latent embeddings (DCAE encoder output)
2. LoRA layer outputs (transformer attention projections)
3. Text prompt embeddings (text encoder output)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
import numpy as np
from collections import defaultdict


class EmbeddingHookManager:
    """Manages PyTorch forward hooks to capture embeddings during training."""

    def __init__(self, callback: Optional[Callable] = None):
        """
        Args:
            callback: Function called with captured embeddings.
                      Signature: callback(space: str, embeddings: np.ndarray, metadata: dict)
        """
        self.callback = callback
        self.hooks = []
        self.current_step = 0
        self.current_batch_prompts = []
        self.enabled = True

    def set_callback(self, callback: Callable):
        self.callback = callback

    def set_step(self, step: int):
        self.current_step = step

    def set_prompts(self, prompts: List[str]):
        self.current_batch_prompts = prompts

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def _make_latent_hook(self, name: str):
        """Hook for VAE/DCAE encoder outputs -- audio latent space."""
        def hook(module, input, output):
            if not self.enabled:
                return
            # dcae.encode() returns (latents, _). The hook fires on the
            # encode sub-module. Output may be a tuple or tensor.
            if isinstance(output, tuple):
                latents = output[0]
            else:
                latents = output
            # Flatten to (batch_size, latent_dim)
            if len(latents.shape) > 2:
                latents = latents.reshape(latents.shape[0], -1)
            latents_np = latents.detach().cpu().float().numpy()
            if self.callback:
                self.callback(
                    space="latent",
                    embeddings=latents_np,
                    metadata={"step": self.current_step, "layer": name},
                )
        return hook

    def _make_lora_hook(self, name: str):
        """Hook for LoRA layer outputs -- transformer attention projections."""
        def hook(module, input, output):
            if not self.enabled:
                return
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if len(out.shape) > 2:
                out = out.reshape(out.shape[0], -1)
            out_np = out.detach().cpu().float().numpy()
            if self.callback:
                self.callback(
                    space="lora",
                    embeddings=out_np,
                    metadata={
                        "step": self.current_step,
                        "layer_name": name,
                    },
                )
        return hook

    def _make_prompt_hook(self, name: str):
        """Hook for text encoder output -- prompt embedding space."""
        def hook(module, input, output):
            if not self.enabled:
                return
            if isinstance(output, tuple):
                embeds = output[0]
            else:
                embeds = output
            # Mean pool over sequence: (batch, seq_len, dim) -> (batch, dim)
            if len(embeds.shape) == 3:
                embeds = embeds.mean(dim=1)
            embeds_np = embeds.detach().cpu().float().numpy()
            if self.callback:
                self.callback(
                    space="prompt",
                    embeddings=embeds_np,
                    metadata={
                        "step": self.current_step,
                        "prompts": list(self.current_batch_prompts),
                    },
                )
        return hook

    def register_hooks(self, model: nn.Module, config: Dict):
        """
        Register hooks on model layers.

        Args:
            model: The Pipeline (LightningModule) instance
            config: Dict with keys 'latent_layers', 'lora_layers', 'prompt_layers'
                    Each value is a list of dot-separated layer name strings.
        """
        self.remove_hooks()

        for layer_name in config.get("latent_layers", []):
            layer = self._get_layer(model, layer_name)
            if layer:
                h = layer.register_forward_hook(self._make_latent_hook(layer_name))
                self.hooks.append(h)
                print(f"[VizHook] Registered latent hook: {layer_name}")

        for layer_name in config.get("lora_layers", []):
            layer = self._get_layer(model, layer_name)
            if layer:
                h = layer.register_forward_hook(self._make_lora_hook(layer_name))
                self.hooks.append(h)
                print(f"[VizHook] Registered LoRA hook: {layer_name}")

        for layer_name in config.get("prompt_layers", []):
            layer = self._get_layer(model, layer_name)
            if layer:
                h = layer.register_forward_hook(self._make_prompt_hook(layer_name))
                self.hooks.append(h)
                print(f"[VizHook] Registered prompt hook: {layer_name}")

        print(f"[VizHook] Total hooks registered: {len(self.hooks)}")

    def _get_layer(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        """Get a layer by dot-separated name (e.g. 'dcae.encoder')."""
        parts = name.split(".")
        layer = model
        try:
            for part in parts:
                if part.isdigit():
                    layer = layer[int(part)]
                else:
                    layer = getattr(layer, part)
            return layer
        except (AttributeError, IndexError, TypeError):
            print(f"[VizHook] Warning: layer '{name}' not found")
            return None

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
