"""
Extract Intermediate Outputs from StreamSpeech
================================================

This script modifies the demo app to save:
1. Source Spanish audio input (WAV file)
2. Discrete speech units output (PT file)

Add this code to your demo/app.py to extract intermediates.
"""

import torch
import soundfile
import os
from datetime import datetime

# Directory to save extracted files - use absolute path
# This file is in demo/, so go up to project root, then into demo/extracted_intermediates
_current_dir = os.path.dirname(os.path.abspath(__file__))
EXTRACT_DIR = os.path.join(_current_dir, "extracted_intermediates")
os.makedirs(EXTRACT_DIR, exist_ok=True)

def save_source_audio(samples, sample_rate, filename_prefix=None, target_sample_rate=16000):
    """
    Save the source audio input as WAV file at 16kHz.
    
    Call this after loading/processing the source audio.
    Location: In demo/app.py, run() function, after samples are loaded
    
    Args:
        samples: numpy array of audio samples
        sample_rate: current sample rate (e.g., 48000)
        filename_prefix: optional prefix for filename
        target_sample_rate: target sample rate (default 16000)
    """
    if filename_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix = f"source_{timestamp}"
    
    output_path = os.path.join(EXTRACT_DIR, f"{filename_prefix}_source_audio_16k.wav")
    
    # Resample to 16kHz if needed
    if sample_rate != target_sample_rate:
        import torch
        samples_tensor = torch.tensor(samples).unsqueeze(0).unsqueeze(0)  # [1, 1, length]
        target_length = int(len(samples) * target_sample_rate / sample_rate)
        samples_tensor = torch.nn.functional.interpolate(
            samples_tensor, size=target_length, mode='linear', align_corners=False
        )
        samples = samples_tensor.squeeze().numpy()
        print(f"  - Resampled from {sample_rate}Hz to {target_sample_rate}Hz")
    
    # Save as WAV file at 16kHz
    soundfile.write(output_path, samples, target_sample_rate)
    
    print(f"✓ Saved source audio: {output_path}")
    print(f"  - Sample rate: {target_sample_rate} Hz (16kHz)")
    print(f"  - Duration: {len(samples)/target_sample_rate:.2f} seconds")
    print(f"  - Shape: {samples.shape}")
    
    return output_path


def save_discrete_units(units_tensor, filename_prefix=None, save_as_text=True):
    """
    Save discrete speech units as PT file (and optionally as text).
    
    Call this when units are generated before being fed to vocoder.
    Location: In agent/speech_to_speech.streamspeech.agent.py, 
              in policy() method, after line 713-724 where units are generated
    
    Args:
        units_tensor: torch tensor of discrete units (can be list or tensor)
        filename_prefix: optional prefix for filename
        save_as_text: also save units as readable text file
    """
    if filename_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix = f"units_{timestamp}"
    
    # Convert to tensor if it's a list
    if isinstance(units_tensor, list):
        units_tensor = torch.tensor(units_tensor, dtype=torch.long)
    
    # Save as PyTorch file
    pt_path = os.path.join(EXTRACT_DIR, f"{filename_prefix}_discrete_units.pt")
    torch.save(units_tensor, pt_path)
    
    print(f"✓ Saved discrete units: {pt_path}")
    print(f"  - Shape: {units_tensor.shape}")
    print(f"  - Number of units: {units_tensor.numel()}")
    print(f"  - Unit range: [{units_tensor.min().item()}, {units_tensor.max().item()}]")
    
    # Also save as text for inspection
    if save_as_text:
        txt_path = os.path.join(EXTRACT_DIR, f"{filename_prefix}_discrete_units.txt")
        units_list = units_tensor.cpu().tolist() if units_tensor.dim() > 0 else [units_tensor.item()]
        with open(txt_path, 'w') as f:
            # Save as space-separated values
            if isinstance(units_list[0], list):
                for row in units_list:
                    f.write(' '.join(map(str, row)) + '\n')
            else:
                f.write(' '.join(map(str, units_list)) + '\n')
        print(f"✓ Saved units as text: {txt_path}")
    
    return pt_path


# Example usage metadata
def save_metadata(source_audio_path, units_path, additional_info=None):
    """Save metadata about the extraction"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_path = os.path.join(EXTRACT_DIR, f"metadata_{timestamp}.txt")
    
    with open(metadata_path, 'w') as f:
        f.write("StreamSpeech Intermediate Outputs\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Source Audio: {source_audio_path}\n")
        f.write(f"Discrete Units: {units_path}\n")
        if additional_info:
            f.write(f"\nAdditional Info:\n")
            for key, value in additional_info.items():
                f.write(f"  {key}: {value}\n")
    
    print(f"✓ Saved metadata: {metadata_path}")
    return metadata_path


"""
INTEGRATION INSTRUCTIONS
========================

1. In demo/app.py, modify the run() function:
   
   Add at line ~859 (after samples are loaded and resampled):
   
   ```python
   # Import the extraction functions
   from extract_intermediates import save_source_audio
   
   # Save source audio at 16kHz
   # Note: Even though samples are at 48kHz here, the function will resample to 16kHz
   save_source_audio(samples, ORG_SAMPLE_RATE, filename_prefix=os.path.basename(source).split('.')[0])
   ```

2. In agent/speech_to_speech.streamspeech.agent.py, modify the policy() method:
   
   Add at line ~744 (before units are fed to vocoder):
   
   ```python
   # Import the extraction functions (add at top of file)
   from demo.extract_intermediates import save_discrete_units
   
   # Save discrete units (add right before line 744)
   if self.states.source_finished:  # Only save final units
       save_discrete_units(unit, filename_prefix="output")
   
   x = {
       "code": torch.tensor(unit, dtype=torch.long, device=self.device).view(
           1, -1
       ),
   }
   ```

3. The extracted files will be saved in: demo/extracted_intermediates/
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nExtracted files will be saved to:", os.path.abspath(EXTRACT_DIR))

