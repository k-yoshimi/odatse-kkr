"""
Command-template utilities used to execute AkaiKKR binaries.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import IO, Mapping, MutableSequence, Optional, Sequence, Union


def as_command_list(command: Union[str, Sequence[str]]) -> Sequence[str]:
    """
    Convert command specifications from TOML into a tokenized list.

    Parameters
    ----------
    command : str or Sequence[str]
        Command specification as a string or list of tokens.

    Returns
    -------
    Sequence[str]
        Tokenized command list.

    Raises
    ------
    ValueError
        If command is neither a string nor an iterable sequence.
    """
    if isinstance(command, str):
        return shlex.split(command)
    try:
        return [str(token) for token in command]
    except TypeError:
        raise ValueError(
            f"command must be a string or iterable sequence, got {type(command).__name__}"
        )


def _replace_placeholders(token: str, input_path: Path, output_path: Path) -> str:
    return (
        token.replace("{input}", input_path.name)
        .replace("{input_path}", str(input_path))
        .replace("{output}", output_path.name)
    )


def _open_relative(path_token: str, work_dir: Path, mode: str) -> IO[str]:
    path = Path(path_token)
    if not path.is_absolute():
        path = work_dir / path
    return path.open(mode)


def run_command_template(
    command_template: Union[str, Sequence[str]],
    *,
    work_dir: Path,
    input_path: Path,
    output_path: Path,
    env: Optional[Mapping[str, str]] = None,
    timeout: Optional[float] = None,
) -> None:
    """
    Expand a command template and execute it via ``subprocess.run``.

    Supports the ``<``/``>`` tokens that redirect stdin/stdout within the work
    directory.  Placeholders ``{input}``, ``{input_path}``, and ``{output}`` are
    expanded before evaluating the tokens.
    """
    tokens = as_command_list(command_template)
    cmd: MutableSequence[str] = []
    stdin_file: Optional[IO[str]] = None
    stdout_file: Optional[IO[str]] = None
    skip_next = False

    try:
        for idx, token in enumerate(tokens):
            if skip_next:
                skip_next = False
                continue

            replaced = _replace_placeholders(token, input_path, output_path)

            if replaced == "<":
                next_idx = idx + 1
                if next_idx < len(tokens):
                    next_token = _replace_placeholders(
                        tokens[next_idx], input_path, output_path
                    )
                    stdin_file = _open_relative(next_token, work_dir, "r")
                    skip_next = True
                else:
                    stdin_file = input_path.open("r")
            elif replaced == ">":
                next_idx = idx + 1
                if next_idx < len(tokens):
                    next_token = _replace_placeholders(
                        tokens[next_idx], input_path, output_path
                    )
                    stdout_file = _open_relative(next_token, work_dir, "w")
                    skip_next = True
                else:
                    stdout_file = output_path.open("w")
            else:
                cmd.append(replaced)

        subprocess.run(
            cmd,
            cwd=work_dir,
            check=True,
            env=env,
            timeout=timeout,
            stdin=stdin_file,
            stdout=stdout_file,
        )
    finally:
        if stdin_file is not None:
            stdin_file.close()
        if stdout_file is not None:
            stdout_file.close()


__all__ = ["as_command_list", "run_command_template"]
