import os
import pandas as pd
from scripts.preprocess.text_pipeline import text_to_phonemes

def generate_manifest(input_csv, output_txt, wav_dir):
    """
    Reads the metadata, phonemizes the text, and writes the VITS training list.
    Format: path/to/wav|phonemes|speaker_id
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    
    # Load metadata (no header, pipe separated)
    df = pd.read_csv(input_csv, sep="|", header=None, names=["id", "text", "norm_text"])
    
    manifest_lines = []
    
    print(f"🔄 Phonemizing {len(df)} lines...")
    
    for _, row in df.iterrows():
        file_id = row["id"]
        text = row["norm_text"]
        
        # 1. Construct the absolute or relative path to the wav
        wav_path = os.path.join(wav_dir, f"{file_id}.wav")
        
        # 2. Get Phonemes
        try:
            phonemes = text_to_phonemes(text)
            
            # 3. Format: wav_path|phonemes|speaker_id
            # Using 0 as the default speaker_id for single-voice training
            line = f"{wav_path}|{phonemes}|0"
            manifest_lines.append(line)
        except Exception as e:
            print(f"❌ Error processing {file_id}: {e}")

    # Write to file
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines))
    
    print(f"✅ Manifest generated at: {output_txt}")
    if manifest_lines:
        print(f"Example line: {manifest_lines[0]}")
    else:
        print("⚠️ Manifest is empty (no lines were successfully phonemized).")

if __name__ == "__main__":
    generate_manifest(
        input_csv="data/raw/metadata.csv",
        output_txt="data/processed/train_list.txt",
        wav_dir="data/raw/wavs"
    )