"""
Utility functions for G1 humanoid RL training and evaluation.

This module provides helper functions for:
  - Model management (loading, saving)
  - Visualization and logging
  - Data processing and statistics
"""

import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


def ensure_dir(path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path.
    
    Returns:
        str: The directory path.
    """
    os.makedirs(path, exist_ok=True)
    return path


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary.
        output_path: Path to save the JSON file.
    """
    ensure_dir(os.path.dirname(output_path))
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4, default=str)
    
    print(f"[INFO] Config saved to: {output_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the JSON config file.
    
    Returns:
        Dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"[INFO] Config loaded from: {config_path}")
    return config


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute basic statistics on a list of values.
    
    Args:
        values: List of numeric values.
    
    Returns:
        Dict: Dictionary with mean, std, min, max, median.
    """
    values = np.array(values)
    
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def format_time(seconds: float) -> str:
    """
    Format seconds into a human-readable time string.
    
    Args:
        seconds: Time in seconds.
    
    Returns:
        str: Formatted time string (e.g., "1h 30m 45s").
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_run_name(prefix: str = "g1") -> str:
    """
    Generate a unique run name with timestamp.
    
    Args:
        prefix: Prefix for the run name.
    
    Returns:
        str: Unique run name.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def print_section(title: str, width: int = 70):
    """
    Print a formatted section header.
    
    Args:
        title: Section title.
        width: Width of the section (including padding).
    """
    border = "=" * width
    padding = (width - len(title) - 2) // 2
    print(f"\n{border}")
    print(f"{' ' * padding} {title}")
    print(f"{border}\n")


def moving_average(values: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    Compute moving average of a 1D array.
    
    Args:
        values: 1D array of values.
        window_size: Size of the moving window.
    
    Returns:
        np.ndarray: Moving average.
    """
    if len(values) < window_size:
        return values
    
    return np.convolve(values, np.ones(window_size)/window_size, mode='valid')


def normalize(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize values to zero mean and unit variance.
    
    Args:
        values: Array to normalize.
        epsilon: Small constant to avoid division by zero.
    
    Returns:
        np.ndarray: Normalized array.
    """
    mean = np.mean(values)
    std = np.std(values)
    return (values - mean) / (std + epsilon)


def clip_actions(actions: np.ndarray, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    """
    Clip actions to a valid range.
    
    Args:
        actions: Action array.
        low: Lower bound.
        high: Upper bound.
    
    Returns:
        np.ndarray: Clipped actions.
    """
    return np.clip(actions, low, high)


class RunLogger:
    """
    Simple logger for tracking training runs.
    
    Attributes:
        run_name: Unique name for this run.
        start_time: When the run started.
        logs: Dictionary of logged metrics.
    """
    
    def __init__(self, run_name: str = None):
        """
        Initialize the logger.
        
        Args:
            run_name: Name for the run; auto-generated if None.
        """
        self.run_name = run_name or get_run_name()
        self.start_time = datetime.now()
        self.logs = {}
    
    def log(self, key: str, value: Any):
        """
        Log a single value.
        
        Args:
            key: Log key.
            value: Log value.
        """
        if key not in self.logs:
            self.logs[key] = []
        self.logs[key].append(value)
    
    def log_dict(self, metrics: Dict[str, Any]):
        """
        Log multiple values at once.
        
        Args:
            metrics: Dictionary of metrics to log.
        """
        for key, value in metrics.items():
            self.log(key, value)
    
    def get_stats(self, key: str) -> Dict[str, float]:
        """
        Get statistics for a logged metric.
        
        Args:
            key: Metric key.
        
        Returns:
            Dict: Statistics for the metric.
        """
        if key not in self.logs:
            raise ValueError(f"Metric '{key}' not found in logs")
        
        return compute_statistics(self.logs[key])
    
    def elapsed_time(self) -> float:
        """
        Get elapsed time since logger creation.
        
        Returns:
            float: Elapsed time in seconds.
        """
        return (datetime.now() - self.start_time).total_seconds()
    
    def summary(self) -> str:
        """
        Generate a summary of all logged metrics.
        
        Returns:
            str: Summary string.
        """
        summary_lines = [f"Run: {self.run_name}"]
        summary_lines.append(f"Elapsed: {format_time(self.elapsed_time())}")
        summary_lines.append("\nMetrics Summary:")
        
        for key in self.logs.keys():
            stats = self.get_stats(key)
            summary_lines.append(
                f"  {key}: mean={stats['mean']:.4f}, "
                f"std={stats['std']:.4f}, "
                f"min={stats['min']:.4f}, max={stats['max']:.4f}"
            )
        
        return "\n".join(summary_lines)
    
    def save(self, output_path: str):
        """
        Save logs to a JSON file.
        
        Args:
            output_path: Path to save the logs.
        """
        ensure_dir(os.path.dirname(output_path))
        
        save_dict = {
            "run_name": self.run_name,
            "start_time": self.start_time.isoformat(),
            "elapsed_seconds": self.elapsed_time(),
            "logs": {k: v for k, v in self.logs.items()},
        }
        
        with open(output_path, 'w') as f:
            json.dump(save_dict, f, indent=4, default=str)
        
        print(f"[INFO] Logs saved to: {output_path}")
