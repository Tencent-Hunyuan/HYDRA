"""
Model utility functions
For printing and analyzing model parameter status
"""
import logging

logger = logging.getLogger(__name__)


def print_parameter_gradient_status(model, logger_instance=None):
    """
    Print gradient status statistics for model parameters

    Args:
        model: a PyTorch model
        logger_instance: a logger instance; if None, the default logger is used
    """
    if logger_instance is None:
        logger_instance = logger

    logger_instance.info("=" * 80)
    logger_instance.info("Model parameter gradient status statistics:")
    logger_instance.info("=" * 80)

    trainable_params = []
    frozen_params = []
    trainable_count = 0
    frozen_count = 0
    trainable_size = 0
    frozen_size = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
            trainable_count += 1
            trainable_size += param.numel()
        else:
            frozen_params.append(name)
            frozen_count += 1
            frozen_size += param.numel()

    logger_instance.info(
        f"Trainable parameters: {trainable_count:,}, total size: {trainable_size:,} "
        f"({trainable_size / 1e6:.2f}M)")
    logger_instance.info(f"Frozen parameters: {frozen_count:,}, total size: {frozen_size:,} ({frozen_size / 1e6:.2f}M)")
    logger_instance.info(
        f"Total parameters: {trainable_count + frozen_count:,}, "
        f"total size: {trainable_size + frozen_size:,} ({(trainable_size + frozen_size) / 1e6:.2f}M)")

    if trainable_size + frozen_size > 0:
        logger_instance.info(f"Trainable ratio: {100 * trainable_size / (trainable_size + frozen_size):.2f}%")

    logger_instance.info("-" * 80)

    if trainable_params:
        logger_instance.info(f"Trainable parameter list (first 20):")
        for name in trainable_params[:2000]:
            logger_instance.info(f"  [trainable] {name}")
        if len(trainable_params) > 20:
            logger_instance.info(f"  ... {len(trainable_params) - 20} more trainable parameters")

    logger_instance.info("-" * 80)
    if frozen_params:
        logger_instance.info(f"Frozen parameter list (first 20):")
        for name in frozen_params[:2000]:
            logger_instance.info(f"  [frozen] {name}")
        if len(frozen_params) > 20:
            logger_instance.info(f"  ... {len(frozen_params) - 20} more frozen parameters")

    logger_instance.info("=" * 80)

    return {
        'trainable_count': trainable_count,
        'frozen_count': frozen_count,
        'trainable_size': trainable_size,
        'frozen_size': frozen_size,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params
    }
