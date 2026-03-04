"""
download_ljspeech_sample.py

Downloads the full LJSpeech-1.1 dataset using torchaudio (which handles
the tarball automatically), then samples N files and writes the metadata
CSV your pipeline expects.

The full dataset (~2.6 GB) is kept in data/raw/LJSpeech-1.1/ — it is
already excluded from git via the data/ entry in .gitignore.

Usage:
    python -m uv run -m scripts.setup.download_ljspeech_sample [--num-samples 20]

Output:
    data/raw/LJSpeech-1.1/wavs/     — full dataset WAVs (kept as-is)
    data/raw/wavs/LJ*.wav           — symlinked / copied subset used for training
    data/raw/metadata.csv           — id|text|normalized_text (pipe-separated)

After this, run:
    python -m uv run -m scripts.preprocess.generate_manifest
"""

import argparse
import os
import shutil

import pandas as pd
import torchaudio

DOWNLOAD_ROOT = os.path.join("data", "raw")           # torchaudio will make LJSpeech-1.1/ here
WAV_OUT_DIR   = os.path.join("data", "raw", "wavs")   # where we copy the subset
META_OUT      = os.path.join("data", "raw", "metadata.csv")


def download_and_sample(num_samples: int = 20) -> None:
    os.makedirs(WAV_OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # torchaudio.datasets.LJSPEECH downloads + extracts automatically.    #
    # If already present it skips the download.                            #
    # ------------------------------------------------------------------ #
    print(f"📥 Downloading LJSpeech-1.1 via torchaudio (skips if already done)...")
    dataset = torchaudio.datasets.LJSPEECH(root=DOWNLOAD_ROOT, download=True)
    print(f"✅ Dataset ready — {len(dataset)} total samples\n")

    # ------------------------------------------------------------------ #
    # Sample the first N entries                                           #
    # ------------------------------------------------------------------ #
    n = min(num_samples, len(dataset))
    print(f"🎯 Sampling first {n} files...\n")

    metadata_rows = []
    ljspeech_wav_dir = os.path.join(DOWNLOAD_ROOT, "LJSpeech-1.1", "wavs")

    for i in range(n):
        # torchaudio LJSPEECH returns: (waveform, sample_rate, transcript, normalized_transcript)
        _, _, transcript, normalized = dataset[i]

        # Derive the file ID — LJSpeech names are 1-indexed: LJ001-0001 …
        # torchaudio doesn't expose the filename directly, but we can get it
        # from the metadata file inside the dataset
        pass  # we'll read from the metadata file directly instead

    # Read the original metadata.csv from the extracted dataset
    orig_meta = os.path.join(DOWNLOAD_ROOT, "LJSpeech-1.1", "metadata.csv")
    df = pd.read_csv(
        orig_meta,
        sep="|",
        header=None,
        names=["id", "text", "normalized_text"],
        quoting=3,  # QUOTE_NONE
    )

    subset = df.head(n)

    print(f"{'ID':<20} {'Text (first 60 chars)'}")
    print("-" * 80)
    for _, row in subset.iterrows():
        src = os.path.join(ljspeech_wav_dir, f"{row['id']}.wav")
        dst = os.path.join(WAV_OUT_DIR, f"{row['id']}.wav")
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        print(f"  {row['id']:<18} {str(row['normalized_text'])[:60]}")

    # Write metadata in your project's pipe-separated format
    subset.to_csv(META_OUT, sep="|", index=False, header=False)

    print(f"\n✅ Done!")
    print(f"   {n} WAV files copied → {WAV_OUT_DIR}/")
    print(f"   Metadata          → {META_OUT}")
    print(f"\nNext step:")
    print(f"   python -m uv run -m scripts.preprocess.generate_manifest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download LJSpeech and sample N files for training"
    )
    parser.add_argument(
        "--num-samples", type=int, default=20,
        help="Number of samples to use (default: 20)"
    )
    args = parser.parse_args()
    download_and_sample(args.num_samples)
