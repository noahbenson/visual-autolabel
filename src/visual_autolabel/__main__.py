# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/__main__.py
# Command-line interface __main__ implementation.

if __name__ == '__main__':
    from .cmd import main
    from sys import (argv, exit)
    exit(main(argv))
