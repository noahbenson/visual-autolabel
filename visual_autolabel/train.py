# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/train.py
# Utilities for training the CNN models of the visual_autolabel library.

"""
The `visual_autolabel.trail` package contains CNN model training utilities for
use in and with the `visual_autolabel` library.
"""


#===============================================================================
# Constants / Globals

#-------------------------------------------------------------------------------
# Logging
# These are strings that encapsulate the printing of training information to the
# terminal or to a log-file.

log_header_format = "%-5s  %-7s  %-5s   %-8s  %-8s  %-8s   %-8s  %-8s  %-8s"
log_format = " | ".join(["%2d/%2d  %7.5f  %5.1f", 
                         "%8.3f  %8.3f  %8.3f",
                         "%8.3f  %8.3f  %8.3f %s"])
log_header = log_header_format % (
    "epoch", "  lr   ", "dt[s]",
    "trn bce ", "trn dice", "trn loss",
    "val bce ", "val dice", "val loss")
log_header_hline = log_header_format % (
    "="*5, "="*7, "="*5, "="*8, "="*8, "="*8, "="*8, "="*8, "="*8)
log_header_hline = log_header_hline.replace(' ', '-')
log_header_hline = "%s+%s+%s" % (
    log_header_hline[:22], log_header_hline[23:53], log_header_hline[54:])


#===============================================================================
# Training Functions

#-------------------------------------------------------------------------------
# Logging

def log_epoch(metrics, epochno=None, epochmax=None, lr=None, dt=None,
              logger=print, endl=""):
    """Logs a single epoch of model-training.

    `log_epoch(metrics, epochno, lr, dt)` logs an epoch's metrics, which must be
    a dict-like object containing the keys `'trn'` and `'val'`, the values for
    each of which must also be a dict-like object whose keys are `'dice'`,
    `'bce'`, and `'loss'`.  If metrics is None, then a header is printed. If
    metrics is Ellipsis, then a separator line is printed.

    Parameters
    ----------
    metrics : dict-like or None or Ellipse
        A dict-like object of the metrics to report.
    epochno : int or None, optional
        The epoch number.
    epochmax : int or None, optional
        The maximum number of epochs that are being run.
    lr : float or None, optional
        The learning-rate of the current epoch.
    dt : float or None, optional
        The amount of time passed during the current epoch.
    logger : function or None, optional
        The logging function to use. If `None`, then nothing is logged. The
        default is `print`.
    endl : str or None, optional
        The end-of-line string to use when printing. For logging functions like
        `print`, which automatically append a newline, this can be `None` or
        just `""`.
    
    Returns
    -------
    None
    """
    if logger is None: return None
    if epochno is None: epochno = 0
    if epochmax is None: epochmax = 0
    if lr is None: lr = 0
    if dt is None: dt = 0
    if endl is None: endl = ""
    if metrics is None:
        logger(log_header + endl)
        logger(log_header_hline + endl)
    elif metrics is Ellipsis:
        logger('-' * len(log_header) + endl)
    else:
        t = metrics['trn']
        v = metrics['val']
        tup = ((epochno+1, epochmax, lr, dt) +
               tuple([t[q] for q in ['bce','dice','loss']]) +
               tuple([v[q] for q in ['bce','dice','loss']]) +
               (endl,))
        logger(log_format % tup)
    return None

#-------------------------------------------------------------------------------
# Training

def train_model(model, optimizer, scheduler, dataloaders,
                num_epochs=10,
                save_path=None,
                device=None,
                hlines=False,
                logger=print,
                endl='',
                logits=None,
                bce_weight=0.5,
                reweight=True,
                smoothing=1):
    """Trains and returns a model based on the various optional arguments.
    
    `train_model(model, optimizer, scheduler, dataloaders)` runs training on the
    given model and returns the newly trained model. This represents a single
    set of epochs with a single `StepLR` decay.

    Parameters
    ----------
    model : PyTorch module
        The PyTorch model to be trained.
    optimizer : PyTorch Optimizer
        The PyTorch optimizer to use for training.
    scheduler : PyTorch Scheduler
        The PyTorch scheduler to be used (for example, PyTorches `StepLR`
        scheduler).
    dataloaders : mapping of DataLoader objects
        A mapping whose keys are `'trn'` and `'val'` and whose values are the
        PyTorch `DataLoader` objects associated with the training and validation
        datasets.
    num_epochs : int, optional
        The number of training epochs to run (default: 10).
    save_path : str or None, optional
        If provided, the model and optimizer state will be pickled to files in
        this directory at the end of every epoch.
    device : str or None, optional
        The device to use for PyTorch training. If `None` (the default), then
        uses CUDA if CUDA is available and otherwise uses CPU.
    hlines : boolean, optional
        Whether to print horizontal lines in the logger for clarity (default:
        False).
    logger : function or None, optional
        The logging function to use. If `None`, then nothing is logged. The
        default is `print`.
    endl : str or None, optional
        The end-of-line string to use when printing. For logging functions like
        `print`, which automatically append a newline, this can be `None` or
        just `""`.
    logits : boolean, optional
        Whether the values in `pred` are logits--i.e., unnormalized scores that
        have not been run through a sigmoid calculation already. If this is
        `True`, then the BCE starts by calculating the sigmoid of the `pred`
        argument. If `None`, then attempts to deduce whether the input is or is
        not logits. The default is `None`.
    bce_weight : float, optional
        The weight to give the BCE-based loss; the weight for the 
        dice-coefficient loss is always `1 - bce_weight`. The default is `0.5`.
    reweight : boolean, optional
        Whether to reweight the classes by calculating the BCE for each class
        then calculating the mean across classes. If `False`, then the raw BCE
        across all pixels, classes, and batches is returned (the default).
    smoothing : number, optional
        The smoothing coefficient `s` to use with the dice-coefficient liss.
        The default is `1`.

    Returns
    -------
    tuple
        A 3-tuple of `(trained_model, best_loss, best_dice_loss)`.
    """
    import time, torch
    from .util import loss as calc_loss
    # Parse some arguments.
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if endl is None:
        endl = ""
    # We need to keep track of the best model weights: start with the current
    # model weights.
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    best_dice = 1e10
    # Log the header.
    log_epoch(None, logger=logger, endl=endl)
    # Now, for each epoch...
    for epoch in range(num_epochs):
        since = time.time()
        allmetrics = {}
        savestr = ""
        lr0 = optimizer.param_groups[0]['lr']
        # Each epoch has a training and validation phase
        for phase in ['trn', 'val']:
            if phase == 'trn':
                model.train()  # Set model to training mode.
            else:
                model.eval()   # Set model to evaluate mode.
            metrics = {}
            epoch_samples = 0
            for (inputs, labels) in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Zero the parameter gradients.
                optimizer.zero_grad()
                # Calculate the forward model.
                with torch.set_grad_enabled(phase == 'trn'):
                    outputs = model(inputs.float())
                    loss = calc_loss(outputs, labels,
                                     logits=logits,
                                     bce_weight=bce_weight,
                                     smoothing=smoothing,
                                     reweight=reweight,
                                     metrics=metrics)
                    # backward + optimize only if in training phase
                    if phase == 'trn':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                # statistics
                epoch_samples += inputs.size(0)
            for k in metrics.keys():
                metrics[k] /= epoch_samples
            epoch_loss = metrics['loss']
            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    savestr = "*"
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                if metrics['dice'] < best_dice:
                    best_dice = metrics['dice']
            allmetrics[phase] = metrics
        time_elapsed = time.time() - since
        log_epoch(allmetrics, epoch, num_epochs, lr0, time_elapsed,
                  endl=savestr, logger=logger)
        if hlines: log_epoch(Ellipsis, logger=logger, endl=endl)
        if save_path is not None:
            torch.save(model.state_dict(), 
                       os.path.join(save_path, "model%06d.pkl" % epoch))
            torch.save(optimizer.state_dict(), 
                       os.path.join(save_path, "optim%06d.pkl" % epoch))
    if logger is not None:
        logger('Best val loss: {:4f}'.format(best_loss) + endl)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return (model, best_loss, best_dice)

