import importlib
import os
import sys
from functools import lru_cache


DEFAULT_LIBRARY = "quantification"
LIBRARY_ENV = "AGENT_LIBRARY"


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def library_name() -> str:
    return os.environ.get(LIBRARY_ENV, DEFAULT_LIBRARY)


def library_root() -> str:
    return os.path.join(project_root(), library_name())


def task_path(filename: str) -> str:
    return os.path.join(library_root(), "task", filename)


def config_reference_path() -> str:
    return task_path("config_reference.yaml")


def reward_interpretation_path() -> str:
    return os.path.join(library_root(), "reward", "interpretation.md")


def configure_library_path() -> str:
    """Add the selected task library folder to sys.path.

    After this, imports such as `from library...` and `import task.task`
    resolve inside the active task library, e.g. `<repo>/quantification/library`.
    """
    root = library_root()
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Task library folder not found: {root}. "
            f"Set {LIBRARY_ENV}=<library_folder_name>."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


@lru_cache(maxsize=None)
def task_prompt():
    configure_library_path()
    return importlib.import_module("task.task")


configure_library_path()

from library.trainer import Trainer
from library.dataio.diffusion import build_dataset
from library.inferer import infer_volume
from library.utility.build_blocks import build_model, build_step_fn, build_volume
