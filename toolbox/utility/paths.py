import os
import shutil
import sys
from typing import Optional


PROJECT_ROOT_ENV = "MICROAGENT_ROOT"


def _looks_like_project_root(path: str) -> bool:
    return (
        os.path.exists(os.path.join(path, "agent.py"))
        and os.path.isdir(os.path.join(path, "quantification"))
        and os.path.isdir(os.path.join(path, "toolbox"))
    )


def _find_project_root_from_cwd() -> Optional[str]:
    path = os.path.abspath(os.getcwd())
    while True:
        if _looks_like_project_root(path):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent


def set_project_root(root: Optional[str] = None) -> str:
    root = root or os.environ.get(PROJECT_ROOT_ENV)
    if root is None:
        root = _find_project_root_from_cwd()
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    root = os.path.abspath(root)
    os.environ[PROJECT_ROOT_ENV] = root
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def get_project_root() -> str:
    return set_project_root(os.environ.get(PROJECT_ROOT_ENV))


def project_path(*parts: str) -> str:
    return os.path.join(get_project_root(), *parts)


def mrtrix_path(command: str) -> str:
    env_var = f"MICROAGENT_{command.upper()}"
    executable = os.environ.get(env_var) or shutil.which(command)
    if executable:
        return os.path.abspath(os.path.expanduser(executable))
    raise FileNotFoundError(
        f"MRtrix3 command '{command}' was not found. Put it on PATH or set "
        f"{env_var} to the executable path."
    )
