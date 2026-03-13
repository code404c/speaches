import shutil
import subprocess
import sys
import time

from faster_whisper import BatchedInferencePipeline, WhisperModel

NVIDIA_SMI = shutil.which("nvidia-smi") or "nvidia-smi"


def get_gpu_memory_mb() -> float:
    try:
        result = subprocess.run(
            [NVIDIA_SMI, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip().split("\n")[0])
    except (subprocess.SubprocessError, ValueError, OSError):
        return -1


def benchmark(audio_path: str, model_id: str = "deepdml/faster-whisper-large-v3-turbo-ct2") -> None:
    print(f"Loading model: {model_id}")
    model = WhisperModel(model_id, device="cuda", device_index=0, compute_type="float16")
    pipeline = BatchedInferencePipeline(model=model)

    batch_sizes = [1, 4, 8, 16, 32]

    print(f"\n{'batch_size':>10} {'time_s':>10} {'gpu_mem_mb':>12}")
    print("-" * 36)

    for bs in batch_sizes:
        mem_before = get_gpu_memory_mb()

        t0 = time.perf_counter()
        segments, info = pipeline.transcribe(audio_path, batch_size=bs)
        _ = list(segments)
        elapsed = time.perf_counter() - t0

        mem_after = get_gpu_memory_mb()
        mem_delta = mem_after - mem_before if mem_before > 0 else -1

        print(f"{bs:>10} {elapsed:>10.2f} {mem_delta:>12.0f}")

    print(f"\nAudio duration: {info.duration:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_batch_size.py <audio_file> [model_id]")
        sys.exit(1)
    audio_file = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "deepdml/faster-whisper-large-v3-turbo-ct2"
    benchmark(audio_file, model)
