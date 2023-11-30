import subprocess
import sys

import pytest

from metatensor_models.scripts import __all__ as available_scripts


class Test_parse_args(object):
    """Tests for argument parsing."""

    def test_required_args(self):
        """Test required arguments."""
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(["metatensor_models"])

    def test_wrong_module(self):
        """Test wrong module."""
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(["metatensor_models", "foo"])

    @pytest.mark.parametrize("module", tuple(available_scripts))
    def test_available_modules(self, module):
        """Test available modules."""
        subprocess.check_call(["metatensor_models", module, "--help"])

    @pytest.mark.parametrize("args", ("version", "help"))
    def test_extra_options(self, args):
        """Test extra options."""
        subprocess.check_call(["metatensor_models", "--" + args])