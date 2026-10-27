"""Compatibility with the current Gripette client and its bundled robot model."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetGraspProjection:
    """Strategy/closure calibration saved with the Grabette training dataset."""

    lim_prox: float = 1.631882850614698
    lim_dist: float = 1.7802358370342162

    def encode(self, proximal: float, distal: float) -> tuple[float, float]:
        return distal / self.lim_dist, proximal / self.lim_prox

    def decode(self, strategy: float, closure: float) -> tuple[float, float]:
        return closure * self.lim_prox, strategy * self.lim_dist


try:
    from gripette.grasp_projection import GraspProjection
except ImportError:
    GraspProjection = DatasetGraspProjection


def resolve_openarm_model(path: str | Path | None = None) -> Path:
    if path is not None:
        result = Path(path).expanduser().resolve()
    else:
        try:
            from openarm_gripette_model import OPENARM_RIGHT_DIR

            result = Path(OPENARM_RIGHT_DIR)
        except ImportError:
            import gripette

            result = Path(gripette.__file__).parent.parent / "urdf/openarm_gripette_right"
    if not (result / "robot.urdf").is_file():
        raise FileNotFoundError(f"OpenArm/Gripette URDF not found at {result}; supply --urdf MODEL_FOLDER")
    return result
