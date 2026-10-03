"""Device selection and run provenance, shared by all three tasks.

Two jobs, both about not being lied to.

**Device selection that can be made strict.** Every task used to call a local
`pick_device()` that walked CUDA -> MPS -> CPU and returned whatever it found. That is
the right behaviour while developing on a laptop and the wrong behaviour on a booked GPU
slot, because the failure mode is silent: a torch wheel built against the wrong CUDA
runtime makes `torch.cuda.is_available()` return `False`, the cascade quietly picks the
CPU, and the only symptom is that the run is thirty times slower than planned. By then
the slot is gone. `--require-cuda` turns that silence into an immediate, explicit error.

**Provenance that is written down.** A metric without the hardware, precision, seed and
source version that produced it cannot be compared against anything, and this project
now has results from two very different devices. Every run emits the same block to both
stdout and a JSON file next to its artifacts, so an MPS number can never be mistaken for
a CUDA number later.

Import from a task script with::

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import runtime as rt
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

# The wheel index every install message points at. cu128 is the floor for the GPU this
# project is scheduled on: an RTX 5090 is Blackwell, sm_120, and the cu124 and cu126
# wheels ship no kernels for it. They install without complaint and then fail at the
# first kernel launch, which is the failure `probe()` exists to catch. A newer index
# works too; an older one does not, regardless of what `nvidia-smi` reports.
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


class CudaUnavailable(RuntimeError):
    """Raised when --require-cuda was given and CUDA cannot actually be used."""


# --------------------------------------------------------------- CLI plumbing

def add_device_args(ap: argparse.ArgumentParser) -> None:
    """The device flags every task shares, so they behave identically everywhere."""
    ap.add_argument(
        "--require-cuda", action="store_true",
        help="fail immediately unless CUDA is present and a kernel actually launches; "
             "use this for every run on the GPU lab PC")
    ap.add_argument(
        "--device", default=None, choices=["cuda", "mps", "cpu"],
        help="force a device instead of auto-detecting; --require-cuda still applies")


# ---------------------------------------------------------- device selection

def probe(device: torch.device) -> str:
    """Actually run a kernel on the device and return a short note, or raise.

    `torch.cuda.is_available()` only means the driver answered the query. It returns
    True in situations where the first real kernel launch then fails -- a driver too old
    for the wheel's compute capability, a wheel with no kernels for the card's
    architecture, an exhausted or exclusive-mode GPU, a container without the device
    node mapped. Since the entire point of `--require-cuda` is to convert a late silent
    failure into an early loud one, the check has to include work the GPU can refuse,
    and it has to catch the refusal rather than letting a raw CUDA traceback out.
    """
    try:
        a = torch.randn(64, 64, device=device)
        out = (a @ a).sum().item()
    except RuntimeError as exc:
        if device.type != "cuda":
            raise
        raise CudaUnavailable(_arch_diagnosis(exc)) from exc
    if out != out:                                   # NaN from a broken kernel
        raise CudaUnavailable(
            f"a matmul on {device} returned NaN, so the device is present but not "
            f"computing correctly")
    return "fp32 matmul ok"


def select_device(require_cuda: bool = False,
                  forced: str | None = None) -> torch.device:
    """CUDA, then Apple MPS, then CPU -- unless told to insist on CUDA.

    `require_cuda` is checked before the cascade runs, not after, so there is no window
    in which the process is already holding CPU tensors it will have to throw away.
    """
    if require_cuda:
        if forced not in (None, "cuda"):
            raise CudaUnavailable(
                f"--require-cuda and --device {forced} contradict each other")
        if not torch.cuda.is_available():
            raise CudaUnavailable(_cuda_diagnosis())
        device = torch.device("cuda")
        probe(device)                                # raises if the GPU refuses work
        return device

    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gpu_arch() -> tuple[str, list[str]]:
    """This GPU's `sm_XX` tag, and the architectures the installed wheel was built for.

    Kept separate from the diagnosis so the provenance block can record both. A results
    file that says "CUDA GPU" but not which architecture cannot explain why the same
    code was fast on one machine and refused to start on another.
    """
    major, minor = torch.cuda.get_device_capability(0)
    return f"sm_{major}{minor}", list(torch.cuda.get_arch_list())


def _arch_diagnosis(exc: Exception) -> str:
    """Why a kernel launch failed on a GPU the driver was happy to report.

    The common cause is a wheel older than the card. It is worth distinguishing from a
    genuine runtime fault because the fix is completely different and the error CUDA
    raises for it ("no kernel image is available for execution on the device") names
    neither the card nor the wheel.
    """
    try:
        sm, arches = gpu_arch()
        name = torch.cuda.get_device_name(0)
    except Exception:                    # the device is too broken even to describe
        return (f"CUDA reported a device but querying it failed:\n  {exc}\n\n"
                "Refusing to fall back to CPU or MPS: see REPRODUCIBILITY.md.")

    lines = [
        f"CUDA is present but the first kernel launch failed on {name}.",
        "",
        f"  torch                {torch.__version__}",
        f"  built against CUDA   {torch.version.cuda}",
        f"  this GPU is          {sm}",
        f"  this wheel targets   {' '.join(arches) or 'nothing listed'}",
    ]
    if sm not in arches:
        lines += [
            "",
            f"{sm} is not in that list, so this wheel ships no kernels for this card.",
            "The wheel is too old for the GPU, which is not something a driver update "
            "fixes. Reinstall from an index new enough to include it:",
            "  pip uninstall -y torch torchvision",
            f"  pip install torch torchvision --index-url {CUDA_INDEX}",
        ]
    else:
        lines += [
            "",
            f"{sm} is in that list, so the wheel does support this card and the cause "
            "is elsewhere. The underlying error was:",
            f"  {exc}",
            "",
            "Check whether another process holds the GPU (`nvidia-smi`), whether it is "
            "in exclusive-compute mode, and whether the driver is older than the "
            f"CUDA {torch.version.cuda} runtime this wheel needs.",
        ]
    lines += ["", "Refusing to fall back to CPU or MPS: see REPRODUCIBILITY.md."]
    return "\n".join(lines)


def _cuda_diagnosis() -> str:
    """Why CUDA is missing, in the terms needed to fix it, not just 'not available'."""
    built = torch.version.cuda
    lines = [
        "--require-cuda was given but torch.cuda.is_available() is False.",
        "",
        f"  torch                {torch.__version__}",
        f"  built against CUDA   {built or 'nothing -- this is a CPU-only build'}",
        f"  platform             {platform.platform()}",
    ]
    if built is None:
        lines += [
            "",
            "This wheel has no CUDA support compiled in. Reinstall from the CUDA index:",
            "  pip uninstall -y torch torchvision",
            f"  pip install torch torchvision --index-url {CUDA_INDEX}",
        ]
    else:
        lines += [
            "",
            "The wheel has CUDA support, so the driver is the likely problem. Check:",
            "  nvidia-smi                      # driver present? CUDA version >= "
            f"{built}?",
            "  echo $CUDA_VISIBLE_DEVICES      # empty string hides every GPU",
        ]
    lines += ["", "Refusing to fall back to CPU or MPS: see REPRODUCIBILITY.md."]
    return "\n".join(lines)


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        return f"Apple MPS ({platform.processor() or 'arm'})"
    return platform.processor() or "cpu"


def device_class(device: torch.device) -> str:
    """"CUDA GPU", "Apple MPS" or "CPU".

    The coarse label, separate from `device_label`'s specific model name. This is the
    field that must never be mixed silently in a results table, so it is a small stable
    vocabulary rather than a free-text string that varies per machine.
    """
    return {"cuda": "CUDA GPU", "mps": "Apple MPS", "cpu": "CPU"}[device.type]


def bf16_ok(device: torch.device) -> bool:
    """Whether bf16 autocast is a real speedup here rather than an emulated slowdown.

    Ampere and later have bf16 tensor cores. `is_bf16_supported()` answers True on some
    older cards where the dtype is emulated, so compute capability is checked too: on
    pre-Ampere the emulation is slower than plain fp32 and buys nothing.
    """
    if device.type != "cuda":
        return False
    if not torch.cuda.is_bf16_supported():
        return False
    return torch.cuda.get_device_capability(0)[0] >= 8


# ----------------------------------------------------------------- seeding

def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every generator the tasks draw from.

    `deterministic` additionally pins cuDNN's algorithm choice. It is off by default
    because it forbids the autotuner and costs real throughput on convolutions, which
    matters for a CycleGAN run sized to a fixed wall-clock budget. Exact
    bit-reproducibility of a GPU run is not something this project claims; seeding the
    generators is, and that is what happens unconditionally here.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


# --------------------------------------------------------------- provenance

def source_version() -> dict:
    """Git commit and dirtiness, or a clear admission that neither is knowable.

    The transfer bundle ships without `.git`, so this returns "not a git checkout" on
    the lab PC rather than pretending. `SOURCE_VERSION` is written into the bundle at
    build time to cover exactly that case.
    """
    stamp = REPO_ROOT / "SOURCE_VERSION"
    try:
        # The toplevel is checked first. Without this, a bundle unzipped *inside* some
        # other git repository -- a home directory under version control, say -- would
        # get that repository's HEAD and record a commit that has nothing to do with the
        # code being run. A wrong provenance record is worse than a missing one.
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=5)
        if top.returncode == 0 and Path(top.stdout.strip()) == REPO_ROOT:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                text=True, timeout=5)
            if commit.returncode == 0:
                dirty = subprocess.run(
                    ["git", "status", "--porcelain"], cwd=REPO_ROOT,
                    capture_output=True, text=True, timeout=5).stdout.strip()
                return {"commit": commit.stdout.strip()[:12],
                        "dirty": bool(dirty),
                        "source": "git"}
    except (OSError, subprocess.SubprocessError):
        pass
    if stamp.exists():
        # Written into the transfer bundle at build time, since the bundle ships without
        # .git but the lab run's provenance still needs to name its source.
        first = stamp.read_text("utf-8").strip().splitlines()[0]
        return {"commit": first.strip()[:12], "dirty": None,
                "source": "SOURCE_VERSION file"}
    return {"commit": None, "dirty": None, "source": "not a git checkout"}


def gpu_memory(device: torch.device) -> dict:
    """Total, allocated and reserved memory, or an explanation of why not.

    Reported separately because they answer different questions: total sizes the batch,
    allocated is what the tensors hold, reserved is what the caching allocator took from
    the driver and will not give back. An out-of-memory error with low allocated and
    high reserved means fragmentation, not a model too large.
    """
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        return {
            "total_mb": round(props.total_memory / 1024**2, 1),
            "allocated_mb": round(torch.cuda.memory_allocated(0) / 1024**2, 1),
            "reserved_mb": round(torch.cuda.memory_reserved(0) / 1024**2, 1),
        }
    if device.type == "mps":
        # MPS shares the system's unified memory, so there is no separate total to
        # report and "GPU memory" is not a meaningful quantity on this hardware.
        return {"total_mb": None,
                "allocated_mb": round(torch.mps.current_allocated_memory() / 1024**2, 1),
                "reserved_mb": None,
                "note": "unified memory; no dedicated GPU pool"}
    return {"total_mb": None, "allocated_mb": None, "reserved_mb": None}


def report(device: torch.device, *, seed: int, batch_size, num_workers,
           precision: str, dataset_sizes: dict, n_params: int | None = None,
           extra: dict | None = None) -> dict:
    """The provenance block. Same keys for every task, every device, every run."""
    block = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
        "device_name": device_label(device),
        # The device *class* is what must never be mixed in a results table, so it is a
        # first-class field rather than something to be inferred from `device_name`.
        "device_class": device_class(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (torch.backends.cudnn.version()
                          if device.type == "cuda" else None),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_capability": (".".join(map(str, torch.cuda.get_device_capability(0)))
                           if device.type == "cuda" else None),
        # Which architectures the installed wheel was compiled for. Recorded because a
        # wheel that does not target this card is the difference between a run that
        # works and one that cannot start, and nothing else in this block would show it.
        "wheel_arch_list": (torch.cuda.get_arch_list()
                            if device.type == "cuda" else None),
        "gpu_memory": gpu_memory(device),
        "precision": precision,
        "seed": seed,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "dataset_sizes": dataset_sizes,
        "n_parameters": n_params,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "source_version": source_version(),
        "env": {k: os.environ[k] for k in
                ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS")
                if k in os.environ},
    }
    if extra:
        block.update(extra)
    return block


def format_report(block: dict) -> list[str]:
    """The provenance block as log lines, for the human reading the terminal."""
    mem = block["gpu_memory"]
    total = f"{mem['total_mb']:,.0f} MB" if mem.get("total_mb") else "n/a (unified)"
    lines = [
        f"runtime device={block['device']} ({block['device_name']}) "
        f"class={block['device_class']}",
        f"runtime torch={block['torch_version']} cuda_available="
        f"{block['cuda_available']} cuda={block['cuda_version']} "
        f"cudnn={block['cudnn_version']} gpus={block['gpu_count']} "
        f"capability={block['gpu_capability']}",
        f"runtime gpu_memory total={total} allocated={mem.get('allocated_mb')} MB "
        f"reserved={mem.get('reserved_mb')} MB",
        f"runtime precision={block['precision']} seed={block['seed']} "
        f"batch_size={block['batch_size']} num_workers={block['num_workers']}",
        f"runtime datasets={json.dumps(block['dataset_sizes'])} "
        f"parameters={block['n_parameters']:,}"
        if block.get("n_parameters") else
        f"runtime datasets={json.dumps(block['dataset_sizes'])}",
        f"runtime python={block['python_version']} platform={block['platform']} "
        f"source={block['source_version']['source']}:"
        f"{block['source_version']['commit']}"
        + (" DIRTY" if block["source_version"]["dirty"] else ""),
    ]
    return lines


def write_report(block: dict, out_dir: Path, tag: str) -> Path:
    """Persist the block beside the run's other artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"runtime_{tag}.json"
    path.write_text(json.dumps(block, indent=2), "utf-8")
    return path


def enforce_cuda_used(device: torch.device, require_cuda: bool) -> None:
    """Assert after the fact that the run really is on CUDA.

    `select_device` already refuses to return anything else, so this is a belt-and-braces
    check for the smoke path, where the requirement is specifically that a passing smoke
    test cannot have quietly run somewhere other than the device the real run will use.
    """
    if require_cuda and device.type != "cuda":
        raise CudaUnavailable(
            f"--require-cuda was given but the run is on {device.type}. A smoke test "
            f"that passes on the wrong device proves nothing about the real run.")


def die(exc: BaseException) -> None:
    """Print a device error the way a person needs to read it, then exit non-zero."""
    print(f"\n{'=' * 74}\nFATAL: {type(exc).__name__}\n{'=' * 74}\n{exc}\n",
          file=sys.stderr)
    raise SystemExit(2)
