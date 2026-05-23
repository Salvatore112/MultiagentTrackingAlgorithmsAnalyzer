from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class TrackingAlgorithm(ABC):
    @abstractmethod
    def run_n_iterations(self, data: Dict[int, Any]) -> Dict[int, Any]:
        pass
