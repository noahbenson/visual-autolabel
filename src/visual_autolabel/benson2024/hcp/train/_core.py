# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/benson2024/hcp/train/_core.py


#===============================================================================
# Dependencies

import os, sys, json

from .._datasets import (
    make_dataloaders)
from .._core import (
    features as features,
    input_properties,
    output_properties)
from ....train import (
    train_until,
    load_training)
from ....util import (
    autolog)
from ....image import (
    UNet)


def main(args, /, exit_on_error=False):
    """Executes the training used by Benson et al. (2024).
    
    This function allows you to run the main function for the command line:
    ```bash
    $ python -m visual_autolabel.benson2024.hcp.train \
                <model_key> <options.json> <plan.json>
    ```
    The function should be called using the following syntax:
    `main([model_key, options_json_filename, plan_json_filename])`.
    """
    # Commandline Arguments.....................................................
    # There must be three of them.
    if len(args) != 3:
        if exit_on_error:
            print(
                "SYNTAX: python -m visual_autolabel.benson2024.hcp.train \\\n"
                "                  <model_key> <options.json> <plan.json>",
                file=sys.stderr)
            sys.exit(1)
        else:
            raise ValueError(f"3 arguments are required; received {len(args)}")
    model_key = args[0]
    opts_filename = os.path.expanduser(os.path.expandvars(args[1]))
    plan_filename = os.path.expanduser(os.path.expandvars(args[2]))
    try:
        with open(opts_filename, 'rt') as fl:
            opts = json.load(fl)
    except Exception as e:
        if exit_on_error:
            print(f"Error reading options file ({opts_filename})\n{str(e)}",
                  file=sys.stderr)
            sys.exit(2)
        else:
            raise
    try:
        with open(plan_filename, 'rt') as fl:
            plan = json.load(fl)
    except Exception as e:
        if exit_on_error:
            print(f"Error reading plan file ({plan_filename})\n{str(e)}",
                  file=sys.stderr)
            sys.exit(2)
        else:
            raise
            
    # Options Parsing...........................................................
    inputs = opts.pop('inputs', None)
    if inputs is None:
        inputs = input_properties
        inputs = dict(inputs)
        del inputs['null']
    elif isinstance(inputs, str):
        if inputs == 'all':
            inputs = input_properties
        elif inputs in input_properties:
            inputs = {inputs: input_properties[inputs]}
        else:
            # Otherwise parse them and then leave them as-is! A dictionary or a
            # list could be provided.
            from ast import literal_eval
            try:
                inputs = literal_eval(inputs)
            except ValueError:
                pass
    
    outputs = opts.pop('prediction', 'area')
    if isinstance(outputs, str):
        outputs = output_properties[outputs]
    # Check if the partition is set to use the default HCP partition.
    if opts.get('partition') == 'default':
        from .._core import partition
        opts['partition'] = partition()


    #===========================================================================
    # Training
    
    # Make an auto-logger with a log-file.
    mcp = opts.get('model_cache_path')
    if mcp is not None:
        log_path = os.path.join(mcp, model_key, 'training.log')
        log = autolog(log_path, clear=True)
    # If we have multiple copies of the model_key, ensure that they match.
    if 'model_key' in opts:
        if model_key == opts['model_key']:
            del opts['model_key']
        
    # Train the model.
    train_until(
        inputs, outputs, plan,
        model_key=model_key,
        model=UNet,
        dataloaders=make_dataloaders,
        features=features,
        logger=log,
        **opts)
