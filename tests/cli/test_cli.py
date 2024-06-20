"""Tests for argument parsing."""

import glob
import shutil
import subprocess
from pathlib import Path
from subprocess import CalledProcessError
from typing import List

import pytest

from metatrain import PACKAGE_ROOT, __version__


COMPFILE = Path(__file__).parents[2] / "src/metatrain/share/metatrain-completion.bash"


def test_required_args():
    """Test required arguments."""
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call(["mtt"])


def test_wrong_module():
    """Test wrong module."""
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call(["mtt", "foo"])


@pytest.mark.parametrize("module", tuple(["eval", "export", "train"]))
def test_available_modules(module):
    """Test available modules."""
    subprocess.check_call(["mtt", module, "--help"])


def test_help():
    subprocess.check_call(["mtt", "--help"])


def test_version():
    stdout = subprocess.run(["mtt", "--version"], capture_output=True).stdout
    assert stdout.decode("ascii") == f"metatrain {__version__}\n"


def test_debug_flag():
    """Test that even if debug flag is set commands run normal."""
    subprocess.check_call(["mtt", "--debug", "train", "-h"])


def test_shell_completion_flag():
    """Test that path to the `shell-completion` is correct."""
    completion_path = subprocess.check_output(["mtt", "--shell-completion"])

    assert Path(completion_path.decode("ascii")).is_file


# TODO: There seems to be an issue with zsh, Github CI and subprocesses.
@pytest.mark.parametrize(
    "shell",
    [
        "bash",
        pytest.param("zsh", marks=pytest.mark.xfail(reason="Github CI - zsh issue")),
    ],
)
def test_syntax_completion(shell):
    """Test that the completion can be sourced"""
    subprocess.check_call(
        args=[
            shutil.which(shell),
            "-i",
            "-c",
            "source $(mtt --shell-completion)",
        ],
    )


def get_completion_suggestions(partial_word: str) -> List[str]:
    """Suggestions of a simulated <tab> completion of a metatrain subcommand.

    https://stackoverflow.com/questions/9137245/unit-test-for-bash-completion-script
    """
    cmd = ["mtt", partial_word]
    cmdline = " ".join(cmd)

    out = subprocess.Popen(
        args=[
            "bash",
            "-i",
            "-c",
            r'source {compfile}; COMP_LINE="{cmdline}" COMP_WORDS=({cmdline}) '
            r"COMP_CWORD={cword} COMP_POINT={cmdlen} $(complete -p {cmd} | "
            r'sed "s/.*-F \\([^ ]*\\) .*/\\1/") && echo ${{COMPREPLY[*]}}'.format(
                compfile=str(COMPFILE),
                cmdline=cmdline,
                cmdlen=len(cmdline),
                cmd=cmd[0],
                cword=cmd.index(partial_word),
            ),
        ],
        stdout=subprocess.PIPE,
    )
    stdout, _ = out.communicate()
    return stdout.decode("ascii").split()


@pytest.mark.parametrize(
    "partial_word, expected_completion",
    [(" ", ["--debug", "--help", "--version", "-h", "eval", "export", "train"])],
)
def test_subcommand_completion(partial_word, expected_completion):
    """Test that expected subcommand completion matches."""
    assert set(get_completion_suggestions(partial_word)) == set(expected_completion)


@pytest.mark.parametrize("subcommand", ["train", "eval"])
def test_error(subcommand, capfd, monkeypatch, tmp_path):
    """Test expected display of errors to stdout and log files."""
    monkeypatch.chdir(tmp_path)

    command = ["mtt", subcommand]
    if subcommand == "eval":
        command += ["model.pt"]

    command += ["foo.yaml"]

    with pytest.raises(CalledProcessError):
        subprocess.check_call(command)

    stdout_log = capfd.readouterr().out

    if subcommand == "train":
        error_glob = glob.glob("outputs/*/*/error.log")
        error_file = error_glob[0]
    else:
        error_file = "error.log"

    error_file = str(Path(error_file).absolute().resolve())

    with open(error_file) as f:
        error_log = f.read()

    assert f"please include the full traceback log from {error_file!r}" in stdout_log
    assert "No such file or directory" in stdout_log
    assert "Traceback" in error_log


def test_run_information(capfd, monkeypatch, tmp_path):
    """Test that run informations are displayed correctly"""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(CalledProcessError):
        subprocess.check_call(["mtt", "export", "experimental.soap_bpnn", "model.ckpt"])

    stdout_log = capfd.readouterr().out

    assert f"Package dir: {PACKAGE_ROOT}" in stdout_log
    assert f"Working dir: {Path('.').absolute()}" in stdout_log
    assert f"Version: {__version__}" in stdout_log
    assert "Command line: mtt export experimental.soap_bpnn model.ckpt" in stdout_log
