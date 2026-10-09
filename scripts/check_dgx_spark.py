"""Validate the installed CUDA stack; optionally download Turbo and synthesize."""

import argparse
import platform
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesize", action="store_true", help="Download the Turbo model and generate speech")
    parser.add_argument("--output", type=Path, default=Path("dgx-spark-smoke.wav"))
    args = parser.parse_args()

    import soundfile as sf
    import torch
    import torchaudio

    print(f"Platform: {platform.system()} {platform.machine()}", flush=True)
    print(f"PyTorch: {torch.__version__}; TorchAudio: {torchaudio.__version__}", flush=True)
    print(f"CUDA runtime: {torch.version.cuda}", flush=True)
    if not torch.__version__.startswith("2.10.0+cu130"):
        raise RuntimeError("Expected torch 2.10.0+cu130. Run scripts/install_dgx_spark.sh.")
    if not torchaudio.__version__.startswith("2.10.0+cu130"):
        raise RuntimeError("Expected matching torchaudio 2.10.0+cu130.")
    if torch.version.cuda != "13.0" or not torch.cuda.is_available():
        raise RuntimeError("CUDA 13.0 is not available. Check nvidia-smi and the NVIDIA driver.")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}", flush=True)
    print(f"Compiled architectures: {torch.cuda.get_arch_list()}", flush=True)

    # Execute kernels and synchronize so unsupported GPU binaries fail here.
    x = torch.ones((128, 128), device="cuda")
    torch.testing.assert_close(x @ x, torch.full_like(x, 128))
    wave = torch.zeros((1, 24000), device="cuda")
    resampled = torchaudio.transforms.Resample(24000, 16000).to("cuda")(wave)
    torch.cuda.synchronize()
    if resampled.shape != (1, 16000):
        raise RuntimeError(f"Unexpected resample shape: {resampled.shape}")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "audio-check.wav"
        sf.write(path, wave.cpu().numpy().T, 24000)
        audio, rate = sf.read(path)
        if rate != 24000 or audio.shape != (24000,):
            raise RuntimeError("WAV round-trip failed.")

    # Import the actual model stack without downloading checkpoints yet.
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    print("CUDA kernels, TorchAudio resampling, WAV I/O and Chatterbox imports passed.", flush=True)
    if args.synthesize:
        model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        wav = model.generate("Hello David. Chatterbox is running on the NVIDIA DGX Spark.")
        if wav.numel() == 0 or not torch.isfinite(wav).all().item():
            raise RuntimeError("Generated audio is empty or contains non-finite samples.")
        sf.write(args.output, wav.detach().cpu().numpy().T, model.sr)
        print(f"Saved {args.output.resolve()} — listen to verify speech quality.")


if __name__ == "__main__":
    main()
