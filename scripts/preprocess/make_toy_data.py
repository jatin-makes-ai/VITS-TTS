import os
import wave
import numpy as np
import pandas as pd

def create_toy_audio(filename, duration=2, sr=22050):
    """Generates a simple sine wave wav file."""
    t = np.linspace(0, duration, int(sr * duration))
    # A simple A440 tone
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)  # Mono
        f.setsampwidth(2)  # 16-bit
        f.setframerate(sr)
        f.writeframes(audio.tobytes())

def setup_toy_dataset(num_samples=10):
    os.makedirs("data/raw/wavs", exist_ok=True)
    
    data = []
    texts = [
        "Hello, this is a test of the emergency broadcast system.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is changing the world of speech.",
        "Low latency is crucial for real time applications.",
        "VITS is an end to end variational inference model."
    ]
    
    for i in range(num_samples):
        file_id = f"toy_{i:03d}"
        wav_path = f"data/raw/wavs/{file_id}.wav"
        create_toy_audio(wav_path)
        
        # Pick a text (rotating)
        text = texts[i % len(texts)]
        data.append([file_id, text, text]) # ID, Text, Normalized Text
        
    df = pd.DataFrame(data)
    df.to_csv("data/raw/metadata.csv", sep="|", index=False, header=False)
    print(f"✅ Created {num_samples} toy samples in data/raw/")

if __name__ == "__main__":
    setup_toy_dataset()