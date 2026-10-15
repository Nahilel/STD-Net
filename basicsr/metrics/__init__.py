from basicsr.metrics.seg_metrics import calculate_drive_metrics, calculate_overlapping_error, overlapping_error
from basicsr.utils.registry import METRIC_REGISTRY

__all__ = [
    "calculate_drive_metrics",
    "calculate_overlapping_error",
    "overlapping_error",
    "calculate_metric",
]


def calculate_metric(data: dict, opt: dict):
    metric_type = opt["type"]
    return METRIC_REGISTRY.get(metric_type)(**data)
