"""
Vocoder Wrapper for StreamSpeech Demo
Provides a unified interface for different vocoder implementations
"""

import torch
import json
from pathlib import Path
from agent.tts.vocoder import CodeHiFiGANVocoderWithDur


class ModifiedVocoderAdapter:
    """
    Adapter for the modified UnitHiFiGAN vocoder with FiLM conditioning.
    Extracts speaker and emotion embeddings from source audio and uses them
    for conditioned speech synthesis.
    """
    
    def __init__(self, checkpoint_path, config_path, device="cpu"):
        """
        Initialize modified vocoder with ECAPA and Emotion2Vec models
        
        Args:
            checkpoint_path: Path to fine-tuned HiFiGAN checkpoint
            config_path: Path to HiFiGAN config JSON
            device: Device to load models on
        """
        self.device = device
        
        # Import model classes
        from models.hifigan_generator import UnitHiFiGANGenerator
        from models.ecapa import ECAPA
        from models.emotion2vec import Emotion2Vec
        
        # Load HiFiGAN config
        with open(config_path, 'r') as f:
            hifigan_config = json.load(f)
        
        # Initialize HiFiGAN generator with FiLM
        self.generator = UnitHiFiGANGenerator(config=hifigan_config, use_film=True).to(device)
        
        # Load checkpoint (prefer ema_generator weights)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "ema_generator" in checkpoint:
            print("✓ Loaded 'ema_generator' weights for modified vocoder")
            self.generator.load_state_dict(checkpoint['ema_generator'])
        elif "generator" in checkpoint:
            print("✓ Loaded 'generator' weights for modified vocoder")
            self.generator.load_state_dict(checkpoint['generator'])
        else:
            print("✓ Loading state_dict directly for modified vocoder")
            self.generator.load_state_dict(checkpoint)
        
        self.generator.eval()
        self.generator.remove_weight_norm()
        
        # Initialize embedding extractors
        print("Loading ECAPA speaker embedding model...")
        self.ecapa = ECAPA(device=device)
        
        print("Loading Emotion2Vec emotion embedding model...")
        self.emotion2vec = Emotion2Vec(device=device)
        
        # Cache for embeddings (per audio file)
        self.speaker_embedding = None
        self.emotion_embedding = None
        self.current_source_audio = None
        
        print("✓ Modified vocoder initialized successfully")
    
    def set_source_audio(self, audio_path):
        """
        Extract and cache speaker/emotion embeddings from source audio.
        Called once per audio file.
        
        Args:
            audio_path: Path to source audio file (str or Path)
        """
        audio_path = Path(audio_path)
        
        # Only extract if it's a new audio file
        if self.current_source_audio != audio_path:
            print(f"Extracting embeddings from: {audio_path.name}")
            
            with torch.no_grad():
                self.speaker_embedding = self.ecapa.extract_speaker_embeddings(
                    wav_path=audio_path
                ).to(self.device)
                
                self.emotion_embedding = self.emotion2vec.extract_emotion_embeddings(
                    wav_path=audio_path
                ).to(self.device)
            
            self.current_source_audio = audio_path
            print(f"✓ Embeddings extracted - Speaker: {self.speaker_embedding.shape}, Emotion: {self.emotion_embedding.shape}")
    
    def __call__(self, x, dur_prediction=False):
        """
        Generate audio from discrete units with speaker/emotion conditioning
        
        Args:
            x: dict with "code" tensor [B, T] containing discrete units
            dur_prediction: Ignored (for interface compatibility)
            
        Returns:
            wav: Generated waveform tensor [T]
            dur: Dummy duration tensor (for interface compatibility)
        """
        if self.speaker_embedding is None or self.emotion_embedding is None:
            raise RuntimeError(
                "Speaker/emotion embeddings not set. Call set_source_audio() first."
            )
        
        units = x["code"]  # [B, T]
        
        with torch.no_grad():
            # Generate audio with conditioning
            # Output shape: [B, 1, L] where L is audio length
            audio_tensor = self.generator(
                units=units,
                speaker=self.speaker_embedding.unsqueeze(0),  # Add batch dim
                emotion=self.emotion_embedding.unsqueeze(0),  # Add batch dim
            )
            
            # Remove batch and channel dimensions: [B, 1, L] -> [L]
            wav = audio_tensor.squeeze()
            
            # Create dummy duration tensor for interface compatibility
            # Assume 320 samples per unit (code_hop_size from config)
            num_units = units.shape[1]
            dur = torch.ones(1, num_units, dtype=torch.long, device=self.device)
        
        return wav, dur


class DualVocoderWrapper:
    """
    Wrapper that runs both original and modified vocoders simultaneously
    for side-by-side comparison
    """
    
    def __init__(self, original_vocoder_path, original_vocoder_cfg, 
                 modified_vocoder_path, modified_vocoder_cfg, device="cpu"):
        """
        Initialize both vocoders
        
        Args:
            original_vocoder_path: Path to original vocoder checkpoint
            original_vocoder_cfg: Path to original vocoder config
            modified_vocoder_path: Path to modified vocoder checkpoint
            modified_vocoder_cfg: Path to modified vocoder config
            device: Device to load vocoders on
        """
        self.device = device
        
        print("Initializing DUAL vocoder system...")
        print("=" * 60)
        
        # Initialize original vocoder
        print("\n[1/2] Loading ORIGINAL vocoder (CodeHiFiGAN)...")
        with open(original_vocoder_cfg) as f:
            vocoder_cfg = json.load(f)
        self.original_vocoder = CodeHiFiGANVocoderWithDur(original_vocoder_path, vocoder_cfg)
        if device == "cuda":
            self.original_vocoder = self.original_vocoder.cuda()
        print("✓ Original vocoder loaded")
        
        # Initialize modified vocoder
        print("\n[2/2] Loading MODIFIED vocoder (UnitHiFiGAN+FiLM)...")
        self.modified_vocoder = ModifiedVocoderAdapter(
            modified_vocoder_path, modified_vocoder_cfg, device
        )
        
        print("\n" + "=" * 60)
        print("✓ DUAL vocoder system ready!")
        print("  - Original: Standard CodeHiFiGAN")
        print("  - Modified: Voice-conditioned UnitHiFiGAN+FiLM")
        print("=" * 60 + "\n")
    
    def set_source_audio(self, audio_path):
        """Extract embeddings for modified vocoder"""
        self.modified_vocoder.set_source_audio(audio_path)
    
    def __call__(self, x, dur_prediction=False):
        """
        Generate audio from both vocoders
        
        Args:
            x: dict with "code" tensor
            dur_prediction: duration prediction flag
            
        Returns:
            dict with 'original' and 'modified' keys containing (wav, dur) tuples
        """
        # Generate with original vocoder
        wav_original, dur_original = self.original_vocoder(x, dur_prediction)
        
        # Generate with modified vocoder
        wav_modified, dur_modified = self.modified_vocoder(x, dur_prediction)
        
        return {
            'original': (wav_original, dur_original),
            'modified': (wav_modified, dur_modified)
        }


class VocoderWrapper:
    """Wrapper to handle multiple vocoder types with different interfaces"""
    
    def __init__(self, vocoder_type, vocoder_path, vocoder_cfg_path, device="cpu"):
        """
        Initialize vocoder wrapper
        
        Args:
            vocoder_type: Type of vocoder ("original" or "modified")
            vocoder_path: Path to vocoder checkpoint
            vocoder_cfg_path: Path to vocoder config file
            device: Device to load vocoder on ("cpu" or "cuda")
        """
        self.vocoder_type = vocoder_type
        self.device = device
        
        if vocoder_type == "original":
            self.vocoder = self._init_original_vocoder(vocoder_path, vocoder_cfg_path)
            # Move to device if CUDA
            if device == "cuda":
                self.vocoder = self.vocoder.cuda()
        elif vocoder_type == "modified":
            # ModifiedVocoderAdapter handles device internally
            self.vocoder = ModifiedVocoderAdapter(vocoder_path, vocoder_cfg_path, device)
        else:
            raise ValueError(
                f"Unknown vocoder type: {vocoder_type}. "
                f"Supported types: 'original', 'modified'"
            )
    
    def _init_original_vocoder(self, vocoder_path, vocoder_cfg_path):
        """Initialize the original CodeHiFiGAN vocoder with duration prediction"""
        with open(vocoder_cfg_path) as f:
            vocoder_cfg = json.load(f)
        return CodeHiFiGANVocoderWithDur(vocoder_path, vocoder_cfg)
    
    def set_source_audio(self, audio_path):
        """
        Set source audio for embedding extraction (modified vocoder only).
        For original vocoder, this is a no-op.
        
        Args:
            audio_path: Path to source audio file
        """
        if self.vocoder_type == "modified":
            self.vocoder.set_source_audio(audio_path)
        # Original vocoder doesn't need source audio
    
    def __call__(self, x, dur_prediction=False):
        """
        Unified interface for vocoder inference
        
        Args:
            x: dict with "code" tensor containing discrete units
            dur_prediction: duration prediction flag
            
        Returns:
            wav: waveform tensor
            dur: duration tensor
        """
        return self.vocoder(x, dur_prediction)

