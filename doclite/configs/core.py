# core system configuration
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

class DUModel(str, Enum):
    LAYOUTLMV3 = "layoutlmv3"
    LILT = "lilt"

class DistillStage(str, Enum): #This enum will later control which losses are active.
    INTERNAL = "internal"
    EXTERNAL = "external"

@dataclass(frozen=True)
class Env:
    ROOT: Path = Path(__file__).resolve().parents[2]  # Path(__file__) converts it into a Path object
    DATA: Path = ROOT / "data"
    PROCESSED: Path = ROOT / "data_processed"
    CHECKPOINTS: Path = ROOT / "checkpoints"
    LOGS: Path = ROOT / "logs"                   # distillation experiments need stable artifact locations.

# Instantiate the environment config so other modules can import ENV.DATA, etc.
ENV = Env()


# Ensure directories exist (safe to run multiple times).
for p in [ENV.DATA, ENV.PROCESSED, ENV.CHECKPOINTS, ENV.LOGS]:
    p.mkdir(parents=True, exist_ok=True)



@dataclass(frozen=True)
class DistillWeights:
    ALPHA_HIDDEN: float = 0
    BETA_ATTN: float = 0
    GAMMA_LOGITS: float = 1.0
    DELTA_TASK: float = 1.0    # corresponds to “weighted loss" components.

# Instantiate weights so other modules can import WEIGHTS.ALPHA_HIDDEN, etc.
WEIGHTS = DistillWeights()