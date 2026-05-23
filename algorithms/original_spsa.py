import numpy as np
from random import random, sample
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from .tracking_algorithm import TrackingAlgorithm


class Original_SPSA(TrackingAlgorithm):
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
        self.beta_1: float = 0.5
        self.beta_2: float = 0.5
        self.beta: float = self.beta_1 + self.beta_2
        self.alpha: float = 0.25
        self.gamma: float = 0.25
        self.b: int = 1
        self.s_norms: Dict[int, float] = {
            i: sum(val * val) for i, val in self.sensors_positions.items()
        }
        self.theta: np.ndarray = np.array(
            [val for _, val in self.true_targets_position.items()]
        )
        self.Delta_abs_value: float = 1 / np.sqrt(self.dimensions)

        self.weight = None

        if adjacency_matrix is None:
            self.adjacency_matrix = [
                [1 if i != j else 0 for j in range(self.number_of_sensors)]
                for i in range(self.number_of_sensors)
            ]
        else:
            self.adjacency_matrix = adjacency_matrix

    def _generate_matrix(self, n: int) -> np.ndarray:
        raw_mat: np.ndarray = np.random.rand(n, n)
        weight: np.ndarray = np.tril(raw_mat) + np.tril(raw_mat, -1).T
        weight = np.around(weight, 1)
        np.fill_diagonal(weight, 0)
        weight = -weight + np.diag([n - 1] * n)
        return weight

    def _condition_number(self, matrix: np.ndarray) -> float:
        eig: np.ndarray = np.linalg.eig(matrix)[0]
        eig_list: List[float] = sorted([abs(n) for n in eig if abs(n) > 0.00001])
        return eig_list[-1] / eig_list[0]

    def _rho_overline(self, meas_1: float, meas_2: float) -> float:
        return meas_1 - meas_2

    def _update_matrix(self) -> Tuple[float, np.ndarray]:
        weight: np.ndarray = self.weight
        cond: float = self._condition_number(weight)
        return cond, weight

    def run_main_algorithm(self) -> Dict[int, Dict[int, np.ndarray]]:
        if self.weight is None:
            self.weight = np.eye(self.number_of_sensors) * (self.number_of_sensors - 1)
            for i in range(self.number_of_sensors):
                for j in range(self.number_of_sensors):
                    if i != j:
                        self.weight[i, j] = -1.0 / (self.number_of_sensors - 1)

        weight = self.weight

        errors: Dict = {}

        theta_hat: Dict[int, Dict[int, np.ndarray]] = {
            target_id: {
                sensor_id: self.init_coords[target_id][sensor_id].copy()
                for sensor_id in self.sensor_ids
            }
            for target_id in self.target_ids
        }

        theta_new: Dict[int, Dict[int, np.ndarray]] = {}
        err: float = 0

        for l in self.target_ids:
            theta_new[l] = {}
            neighbors: Dict[int, List[int]] = self._get_random_neibors(weight, 2)

            for i in self.sensor_ids:
                coef1: int = 1 if random() < 0.5 else -1
                coef2: int = 1 if random() < 0.5 else -1
                delta: np.ndarray = np.array(
                    [coef1 * self.Delta_abs_value, coef2 * self.Delta_abs_value]
                )

                x1: np.ndarray = theta_hat[l][i] + self.beta_1 * delta
                x2: np.ndarray = theta_hat[l][i] - self.beta_2 * delta

                y1: float = self._f_l_i(l, i, x1, neighbors)
                y2: float = self._f_l_i(l, i, x2, neighbors)

                spsa: np.ndarray = (y1 - y2) / self.beta * delta / 2

                neighbors_i: List[int] = neighbors.get(i, [])
                b: np.ndarray = weight[i]

                if len(neighbors_i) > 0:
                    theta_diff: List[np.ndarray] = [
                        abs(b[j]) * (theta_hat[l][i] - theta_hat[l][j])
                        for j in neighbors_i
                    ]
                    theta_new[l][i] = theta_hat[l][i] - (
                        self.alpha * spsa + self.gamma * sum(theta_diff)
                    )
                else:
                    theta_new[l][i] = theta_hat[l][i] - self.alpha * spsa

                err += self._compute_error(
                    theta_new[l][i], self.true_targets_position[l]
                )

        theta_hat = theta_new.copy()

        self.errors = errors

        target_err: Dict[int, Dict[int, float]] = {}
        for target_id in self.target_ids:
            target_err[target_id] = {
                sensor_id: self._compute_error(
                    theta_hat[target_id][sensor_id],
                    self.true_targets_position[target_id],
                )
                for sensor_id in self.sensor_ids
            }

        return theta_hat

    def _f_l_i(
        self, l: int, i: int, r_hat_l: np.ndarray, neibors: Dict[int, List[int]]
    ) -> float:
        C: np.ndarray = self._C_i(i, neibors)
        D: List[float] = self._D_l_i(l, i, neibors)

        if C.shape[0] == 0:
            return float(np.linalg.norm(r_hat_l - self.sensors_positions[i]) ** 2)

        if C.shape[0] == 1:
            C_i_inv = 1.0 / C[0, 0] if C[0, 0] != 0 else 1.0
            diff = r_hat_l - (C_i_inv * D[0])
            return float(sum(diff * diff))

        try:
            C_i_inv: np.ndarray = np.linalg.inv(C)
        except Exception:
            C_i_inv = np.linalg.pinv(C)

        diff: np.ndarray = r_hat_l - np.matmul(C_i_inv, D)
        return float(sum(diff * diff))

    def _C_i(self, i: int, neibors: Dict[int, List[int]]) -> np.ndarray:
        neighbor_list = neibors.get(i, [])
        C_i: List[np.ndarray] = [
            self.sensors_positions.get(j) - self.sensors_positions.get(i)
            for j in neighbor_list
        ]
        if len(C_i) == 0:
            return np.array([])
        return 2 * np.array(C_i)

    def _D_l_i(self, l: int, i: int, neibors: Dict[int, List[int]]) -> List[float]:
        neighbor_list = neibors.get(i, [])
        Dli: List[float] = [
            self._calc_D_l_i_j(self.distances.get(l), i, j) for j in neighbor_list
        ]
        return Dli

    def _calc_D_l_i_j(self, meas_l: Dict[int, float], i: int, j: int) -> float:
        return (
            self._rho_overline(meas_l.get(i), meas_l.get(j))
            + self.s_norms.get(j)
            - self.s_norms.get(i)
        )

    def _gen_new_coordinates(self, coords: np.ndarray, R: float = 1) -> np.ndarray:
        phi: float = 2 * np.pi * random()
        rad: float = R * random()

        shift: np.ndarray = rad * np.array([np.sin(phi), np.cos(phi)])
        return coords + shift

    def _compute_error(self, vector_1: np.ndarray, vector_2: np.ndarray) -> float:
        return float(pow(sum(vector_1 - vector_2), 2))

    def _get_random_neibors(
        self, weight: np.ndarray, max: int = 1
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
            if len(neib) > max:
                neib = sample(neib, max)
            neibors[sensor_id] = neib

        return neibors

    def run_n_iterations(self, data: Dict[int, Any]) -> Dict[int, Any]:
        result: Dict[int, Any] = defaultdict()
        for iteration in data.keys():
            self.true_targets_position = data[iteration][0]
            self.distances = data[iteration][1]
            new_estimates: Dict[int, Dict[int, np.ndarray]] = self.run_main_algorithm()
            self.init_coords = new_estimates
            result[iteration] = [data[iteration][0], new_estimates]
        return result
