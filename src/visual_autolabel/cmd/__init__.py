# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/cmd/__init__.py
# Command-line interface module.

"""Command-line interface for the visual_autolabel library.

The command-line module includes the function `main(argv)` which accepts a list
of arguments `argv` that should be treated the same as `sys.argv`.

The module also includes an `__main__` implementation, so it can be executed
directly via `python -m visual_autolabel.cmd <args...>`.
"""

from ._core import main


