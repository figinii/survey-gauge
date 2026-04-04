from runstats import Statistics
from typing import List
import math

class ScoreTracker():
  def __init__(self, number_of_buckets: int, objective: float):
    """
    Initialize score tracker with per-bucket statistics.
    
    Args:
        number_of_buckets: Number of buckets to track
        objective: Standard error of the mean threshold for convergence
    """
    self.archive = [Statistics() for _ in range(number_of_buckets)]
    self.number_of_buckets = number_of_buckets
    self.objective = objective
  
  def update(self, samples: List[float]) -> None:
    """
    Update means and standard errors with new samples.
    
    Args:
        samples: List of n_buckets numbers representing a sample taken
        
    Raises:
        ValueError: If samples length doesn't match number_of_buckets
    """
    if len(samples) != self.number_of_buckets:
      raise ValueError(f"Expected {self.number_of_buckets} samples, got {len(samples)}")
    
    for i, sample in enumerate(samples):
      self.archive[i].push(sample)
  
  def _compute_sem(self, stats: Statistics) -> float:
    """
    Compute the standard error of the mean (SEM).
    
    Args:
        stats: Statistics object
        
    Returns:
        Standard error of the mean: stdev / sqrt(n)
    """
    if len(stats) < 2:
      return float('inf')
    return stats.stddev() / math.sqrt(len(stats))
  
  def get_underdetermined_indexes(self) -> list:
    """
    Returns indexes of buckets that have NOT yet converged (high uncertainty).
    These are buckets where standard error of the mean >= objective.
    
    Returns:
        List of indexes where SEM is >= objective
    """
    return [i for i, stats in enumerate(self.archive) 
            if self._compute_sem(stats) >= self.objective]
