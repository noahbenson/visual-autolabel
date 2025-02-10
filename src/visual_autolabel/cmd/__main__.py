# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/cmd/__main__.py
# Command-line interface __main__ implementation.

if __name__ == '__main__':
    from ._core import main
    from sys import (argv, exit)
    exit(main(argv))
