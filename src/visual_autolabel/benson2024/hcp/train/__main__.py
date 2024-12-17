# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/benson2024/hcp/train/__main__.py

if __name__ == '__main__':
    from sys import argv, exit
    from ._core import main
    main(argv[1:])
    # Exit nicely.
    exit(0)
