# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/benson2024/hcp/train/__init__.py

"""Namespace that includes API code for reproducing or reusing the training
pipeline of Benson et al. (2024).

This namespace primarily exists to support a command-line API:
```bash
$ python -m visual_autolabel.benson2024.hcp.train \
            <model_key> <options.json> <plan.json>
```

The `main` function in this namespace can be used to invoke this command-line
from Python.
"""

from ._core import main
