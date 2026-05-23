import numpy as np
from random import sample
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from .tracking_algorithm import TrackingAlgorithm


class Distributed_Kalman_Filter(TrackingAlgorithm):
    def __init__(
        self,
        sensors_positions: Optional[Dict[int, np.ndarray]] = None,
        true_targets_position: Optional[Dict[int, np.ndarray]] = None,
        distances: Optional[Dict[int, Dict[int, float]]] = None,
        init_coords: Optional[Dict[int, Dict[int, np.ndarray]]] = None,
        adjacency_matrix: Optional[List[List[int]]] = None,
    ) -> None:
        self.number_of_sensors: int = len(sensors_positions.items())
        self.sensor_ids: Set[int] = {i for i in range(self.number_of_sensors)}
        self.sensors_positions: Optional[Dict[int, np.ndarray]] = sensors_positions
        self.number_of_targets: int = len(true_targets_position.keys())
        self.target_ids: Set[int] = {i for i in range(self.number_of_targets)}
        self.true_targets_position: Optional[Dict[int, np.ndarray]] = (
            true_targets_position
        )
        self.distances: Optional[Dict[int, Dict[int, float]]] = distances
        self.init_coords: Optional[Dict[int, Dict[int, np.ndarray]]] = init_coords

        self.dimensions: int = 2
        self.F = np.eye(self.dimensions)
        self.H = np.eye(self.dimensions)
        self.Q = np.eye(self.dimensions) * 0.01
        self.R = np.eye(self.dimensions) * 0.1

        self.weight = None

        if adjacency_matrix is None:
            self.adjacency_matrix = [
                [1 if i != j else 0 for j in range(self.number_of_sensors)]
                for i in range(self.number_of_sensors)
            ]
        else:
            self.adjacency_matrix = adjacency_matrix

    def _init_weight(self):
        self.weight = np.eye(self.number_of_sensors) * (self.number_of_sensors - 1)
        for i in range(self.number_of_sensors):
            for j in range(self.number_of_sensors):
                if i != j:
                    self.weight[i, j] = -1.0 / (self.number_of_sensors - 1)

    def _get_random_neibors(
        self, weight: np.ndarray, max_n: int = 2
    ) -> Dict[int, List[int]]:
        neibors_mat: np.ndarray = (weight != 0).astype(int)
        np.fill_diagonal(neibors_mat, 0)

        for i in range(self.number_of_sensors):
            for j in range(self.number_of_sensors):
                if self.adjacency_matrix[i][j] == 0:
                    neibors_mat[i, j] = 0

        neibors: Dict[int, List[int]] = {}
        for sensor_id in self.sensor_ids:
            neib: List[int] = [
                ind for ind, sens in enumerate(neibors_mat[sensor_id]) if sens == 1
            ]
            if len(neib) > max_n:
                neib = sample(neib, max_n)
            neibors[sensor_id] = neib
        return neibors

    def _compute_error(self, vector_1: np.ndarray, vector_2: np.ndarray) -> float:
        return float(pow(sum(vector_1 - vector_2), 2))

    def run_main_algorithm(self) -> Dict[int, Dict[int, np.ndarray]]:
        if self.weight is None:
            self._init_weight()

        weight = self.weight

        theta_hat: Dict[int, Dict[int, np.ndarray]] = {
            l: {i: self.init_coords[l][i].copy() for i in self.sensor_ids}
            for l in self.target_ids
        }

        P: Dict[int, Dict[int, np.ndarray]] = {
            l: {i: np.eye(self.dimensions) for i in self.sensor_ids}
            for l in self.target_ids
        }

        theta_new: Dict[int, Dict[int, np.ndarray]] = {}

        for l in self.target_ids:
            theta_new[l] = {}
            neighbors = self._get_random_neibors(weight, 2)

            for i in self.sensor_ids:
                x_pred = self.F @ theta_hat[l][i]
                P_pred = self.F @ P[l][i] @ self.F.T + self.Q

                z = self.true_targets_position[l]

                S = self.H @ P_pred @ self.H.T + self.R
                K = P_pred @ self.H.T @ np.linalg.inv(S)

                x_upd = x_pred + K @ (z - self.H @ x_pred)
                P_upd = (np.eye(self.dimensions) - K @ self.H) @ P_pred

                neighbors_i = neighbors.get(i, [])
                if len(neighbors_i) > 0:
                    consensus = sum(theta_hat[l][j] for j in neighbors_i) / len(
                        neighbors_i
                    )
                    x_upd = 0.5 * x_upd + 0.5 * consensus

                theta_new[l][i] = x_upd
                P[l][i] = P_upd

        return theta_new

    def run_n_iterations(self, data: Dict[int, Any]) -> Dict[int, Any]:
        result: Dict[int, Any] = defaultdict()
        for iteration in data.keys():
            self.true_targets_position = data[iteration][0]
            self.distances = data[iteration][1]
            new_estimates = self.run_main_algorithm()
            self.init_coords = new_estimates
            result[iteration] = [data[iteration][0], new_estimates]
        return result
