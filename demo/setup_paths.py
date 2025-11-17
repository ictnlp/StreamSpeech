#!/usr/bin/env python3
"""
StreamSpeech Paths Setup Script

This script helps you set up the paths_config.json file for your environment.
Run this script to automatically generate the paths configuration.
"""

import os
import json
import sys
from pathlib import Path

def get_streamspeech_root():
    """Get the StreamSpeech root directory"""
    current_dir = Path(__file__).parent.parent.absolute()
    return str(current_dir).replace('\\', '/')

def setup_paths():
    """Set up paths configuration"""
    streamspeech_root = get_streamspeech_root()
    
    # Default paths based on current directory structure
    paths_config = {
        "_comment": "StreamSpeech Paths Configuration - Auto-generated",
        "_note": "Use forward slashes (/) for paths, even on Windows",
        
        "streamspeech_root": streamspeech_root,
        "pretrain_models_root": f"{streamspeech_root}/pretrain_models",
        
        "language_pair": "es-en",
        
        "models": {
            "simultaneous": f"{streamspeech_root}/pretrain_models/streamspeech.simultaneous.es-en.pt",
            "offline": f"{streamspeech_root}/pretrain_models/streamspeech.offline.es-en.pt"
        },
        
        "vocoder": {
            "checkpoint": f"{streamspeech_root}/pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000",
            "config": f"{streamspeech_root}/pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/config.json"
        },
        
        "configs": {
            "data_bin": f"{streamspeech_root}/configs/es-en",
            "user_dir": f"{streamspeech_root}/researches/ctc_unity",
            "agent_dir": f"{streamspeech_root}/agent"
        },
        
        "demo": {
            "upload_folder": f"{streamspeech_root}/demo/uploads",
            "host": "0.0.0.0",
            "port": 7860
        }
    }
    
    # Check if files exist
    print("Checking if required files exist...")
    missing_files = []
    
    for key, path in [
        ("Simultaneous Model", paths_config["models"]["simultaneous"]),
        ("Offline Model", paths_config["models"]["offline"]),
        ("Vocoder Checkpoint", paths_config["vocoder"]["checkpoint"]),
        ("Vocoder Config", paths_config["vocoder"]["config"]),
        ("Data Bin", paths_config["configs"]["data_bin"]),
        ("User Dir", paths_config["configs"]["user_dir"]),
        ("Agent Dir", paths_config["configs"]["agent_dir"])
    ]:
        if os.path.exists(path):
            print(f"✅ {key}: {path}")
        else:
            print(f"❌ {key}: {path} (NOT FOUND)")
            missing_files.append((key, path))
    
    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} files/directories are missing!")
        print("Please ensure all models are downloaded and paths are correct.")
        response = input("Do you want to continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return False
    
    # Write the configuration file
    config_path = Path(__file__).parent / "paths_config.json"
    with open(config_path, 'w') as f:
        json.dump(paths_config, f, indent=4)
    
    print(f"\n✅ Paths configuration saved to: {config_path}")
    print("You can now run the StreamSpeech demo!")
    
    return True

if __name__ == "__main__":
    print("StreamSpeech Paths Setup")
    print("=" * 30)
    
    if setup_paths():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Activate your virtual environment: streamspeech_env\\Scripts\\activate")
        print("2. Run the demo: python app.py")
        print("3. Open your browser to: http://localhost:7860")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
