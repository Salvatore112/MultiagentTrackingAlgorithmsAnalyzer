import numpy as np
import matplotlib.colors as mcolors
import os
import random
import math
import matplotlib.pyplot as plt
import pickle
import hashlib
import time

from enum import Enum
from matplotlib.animation import FuncAnimation, PillowWriter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any


class TARGET_TYPE(Enum):
    LINEAR = 1
    RANDOM_WALK = 2

    def __str__(self) -> str:
        if self == TARGET_TYPE.LINEAR:
            return "lin"
        elif self == TARGET_TYPE.RANDOM_WALK:
            return "ran_wlk"
        return self.name


@dataclass
class Target:
    id: int
    initial_position: Tuple[float, float]
    velocity: Tuple[float, float]
    movement_type: TARGET_TYPE = TARGET_TYPE.LINEAR
    random_walk_params: Optional[Dict[str, float]] = None
    unique_hash: Optional[str] = None


@dataclass
class Sensor:
    id: int
    position: Tuple[float, float]


class Simulation:
    def __init__(
        self,
        duration: float,
        time_step: float = 1.0,
        output_dir: str = "simulation_results",
        noise_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.duration: float = duration
        self.time_step: float = time_step
        self.sensors: List[Sensor] = []
        self.targets: List[Target] = []
        self.simulation_data: Dict[str, Any] = {}
        self.output_dir: str = output_dir
        self.noise_config: Dict[str, Any] = noise_config or {}
        os.makedirs(self.output_dir, exist_ok=True)

    def add_sensor(self, sensor_id: int, position: Tuple[float, float]) -> None:
        self.sensors.append(Sensor(sensor_id, position))

    def add_target(
        self,
        obj_id: int,
        initial_position: Tuple[float, float],
        velocity: Tuple[float, float],
        movement_type: TARGET_TYPE = TARGET_TYPE.LINEAR,
        random_walk_params: Optional[Dict[str, float]] = None,
    ) -> None:
        unique_hash: str = hashlib.sha256(
            f"{obj_id}_{initial_position}_{velocity}_{movement_type}_{random_walk_params}".encode()
        ).hexdigest()
        self.targets.append(
            Target(
                obj_id,
                initial_position,
                velocity,
                movement_type,
                random_walk_params,
                unique_hash,
            )
        )

    def add_linear_target(
        self, obj_id: int, area_size: float = 50
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        timestamp: float = time.time()
        seed: str = hashlib.sha256(f"linear_{obj_id}_{timestamp}".encode()).hexdigest()
        random_state: Any = random.getstate()
        random.seed(seed)

        initial_x: float = random.uniform(-area_size, area_size)
        initial_y: float = random.uniform(-area_size, area_size)
        speed: float = random.uniform(0.5, 3.0)
        angle: float = random.uniform(0, 2 * math.pi)
        vx: float = speed * math.cos(angle)
        vy: float = speed * math.sin(angle)

        random.setstate(random_state)

        self.targets.append(
            Target(
                obj_id, (initial_x, initial_y), (vx, vy), TARGET_TYPE.LINEAR, None, seed
            )
        )
        return (initial_x, initial_y), (vx, vy)

    def add_random_walk_target(
        self, obj_id: int, area_size: float = 50
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        timestamp: float = time.time()
        seed: str = hashlib.sha256(f"random_{obj_id}_{timestamp}".encode()).hexdigest()
        random_state: Any = random.getstate()
        random.seed(seed)

        initial_x: float = random.uniform(-area_size, area_size)
        initial_y: float = random.uniform(-area_size, area_size)
        speed: float = random.uniform(0.5, 2.0)
        angle: float = random.uniform(0, 2 * math.pi)
        vx: float = speed * math.cos(angle)
        vy: float = speed * math.sin(angle)
        random_walk_params: Dict[str, float] = {
            "speed_variation": random.uniform(0.1, 0.5),
            "direction_change_prob": random.uniform(0.1, 0.3),
            "max_direction_change": math.pi / 4,
        }

        random.setstate(random_state)

        self.targets.append(
            Target(
                obj_id,
                (initial_x, initial_y),
                (vx, vy),
                TARGET_TYPE.RANDOM_WALK,
                random_walk_params,
                seed,
            )
        )
        return (initial_x, initial_y), (vx, vy)

    def add_uniform_sensor(
        self, sensor_id: int, area_size: float = 50
    ) -> Tuple[float, float]:
        seed: str = hashlib.sha256(f"sensor_{sensor_id}".encode()).hexdigest()
        random_state: Any = random.getstate()
        random.seed(seed)

        pos_x: float = random.uniform(-area_size, area_size)
        pos_y: float = random.uniform(-area_size, area_size)

        random.setstate(random_state)

        self.sensors.append(Sensor(sensor_id, (pos_x, pos_y)))
        return (pos_x, pos_y)

    def calculate_distance(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        return (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2

    def get_target_position(self, obj: Target, time: float) -> Tuple[float, float]:
        if obj.movement_type == TARGET_TYPE.LINEAR:
            return self._get_linear_position(obj, time)
        elif obj.movement_type == TARGET_TYPE.RANDOM_WALK:
            return self._get_random_walk_position(obj, time)
        else:
            return self._get_linear_position(obj, time)

    def _get_linear_position(self, obj: Target, time: float) -> Tuple[float, float]:
        x: float = obj.initial_position[0] + obj.velocity[0] * time
        y: float = obj.initial_position[1] + obj.velocity[1] * time
        return (x, y)

    def _get_random_walk_position(
        self, obj: Target, time: float
    ) -> Tuple[float, float]:
        random_state: Any = random.getstate()
        random.seed(obj.unique_hash)

        time_points: np.ndarray = np.arange(0, time + self.time_step, self.time_step)
        x: float = obj.initial_position[0]
        y: float = obj.initial_position[1]
        current_vx: float = obj.velocity[0]
        current_vy: float = obj.velocity[1]
        params: Dict[str, float] = obj.random_walk_params or {
            "speed_variation": 0.3,
            "direction_change_prob": 0.2,
            "max_direction_change": math.pi / 4,
        }

        for t in time_points[1:]:
            if random.random() < params["direction_change_prob"]:
                angle_change: float = random.uniform(
                    -params["max_direction_change"], params["max_direction_change"]
                )
                current_speed: float = math.sqrt(current_vx**2 + current_vy**2)
                current_angle: float = math.atan2(current_vy, current_vx)
                new_angle: float = current_angle + angle_change

                speed_variation: float = random.uniform(
                    1 - params["speed_variation"], 1 + params["speed_variation"]
                )
                new_speed: float = current_speed * speed_variation

                current_vx = new_speed * math.cos(new_angle)
                current_vy = new_speed * math.sin(new_angle)

            x += current_vx * self.time_step
            y += current_vy * self.time_step

        random.setstate(random_state)

        return (x, y)

    def run_simulation(self) -> None:
        time_points: np.ndarray = np.arange(
            0, self.duration + self.time_step, self.time_step
        )

        self.simulation_data = {
            "time_points": time_points,
            "sensors": {sensor.id: sensor for sensor in self.sensors},
            "targets": {target.id: target for target in self.targets},
        }

    def get_distance(self, sensor_id: int, target_id: int, time: float) -> float:
        if time < 0 or time > self.duration:
            raise ValueError(
                f"Time {time} is outside the simulation range [0, {self.duration}]"
            )

        sensor: Sensor = self.simulation_data["sensors"][sensor_id]
        target_obj: Target = self.simulation_data["targets"][target_id]

        target_pos: Tuple[float, float] = self.get_target_position(target_obj, time)
        distance: float = self.calculate_distance(sensor.position, target_pos)

        return distance

    def get_target_position_at_time(
        self, target_id: int, time: float
    ) -> Tuple[float, float]:
        if time < 0 or time > self.duration:
            raise ValueError(
                f"Time {time} is outside the simulation range [0, {self.duration}]"
            )

        target_obj: Target = self.simulation_data["targets"][target_id]
        return self.get_target_position(target_obj, time)

    def print_distances_at_time(self, time: float) -> None:
        print(f"\n=== Distances at time t={time} seconds ===")
        for sensor in self.sensors:
            for obj in self.targets:
                distance: float = self.get_distance(sensor.id, obj.id, time)
                obj_pos: Tuple[float, float] = self.get_target_position_at_time(
                    obj.id, time
                )
                print(f"Sensor {sensor.id} -> Target {obj.id}: {distance:.2f} units")
                print(
                    f"  Target {obj.id} position: ({obj_pos[0]:.1f}, {obj_pos[1]:.1f})"
                )

    def print_multiple_times(self, times: List[float]) -> None:
        for time in times:
            self.print_distances_at_time(time)

    def plot_trajectories(self, save_file: bool = True) -> None:
        plt.figure(figsize=(12, 10))

        obj_colors: List[str] = list(mcolors.TABLEAU_COLORS.keys())
        sensor_colors: List[str] = ["red", "green", "blue", "purple", "orange", "brown"]

        for i, obj in enumerate(self.targets):
            color: str = obj_colors[i % len(obj_colors)]

            positions: List[Tuple[float, float]] = [
                self.get_target_position(obj, t)
                for t in self.simulation_data["time_points"]
            ]
            x_vals: List[float] = [p[0] for p in positions]
            y_vals: List[float] = [p[1] for p in positions]

            plt.plot(
                x_vals,
                y_vals,
                color=color,
                linewidth=2,
                label=f"Target {obj.id} ({obj.movement_type})",
                alpha=0.7,
            )

            plt.scatter(
                x_vals[0],
                y_vals[0],
                color=color,
                s=100,
                marker="o",
                edgecolors="black",
                zorder=5,
                label=f"Target {obj.id} start",
            )
            plt.scatter(
                x_vals[-1],
                y_vals[-1],
                color=color,
                s=100,
                marker="s",
                edgecolors="black",
                zorder=5,
                label=f"Target {obj.id} finish",
            )

            if len(positions) > 1 and obj.movement_type == TARGET_TYPE.LINEAR:
                mid_idx: int = len(positions) // 2
                plt.annotate(
                    "",
                    xy=positions[mid_idx + 1],
                    xytext=positions[mid_idx],
                    arrowprops=dict(arrowstyle="->", color=color, lw=2),
                )

        for i, sensor in enumerate(self.sensors):
            color: str = sensor_colors[i % len(sensor_colors)]
            plt.scatter(
                sensor.position[0],
                sensor.position[1],
                color=color,
                s=200,
                marker="^",
                label=f"Sensor {sensor.id}",
                edgecolors="black",
                zorder=5,
            )

            plt.annotate(
                f"S{sensor.id}",
                (sensor.position[0], sensor.position[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
            )

        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.title("Target trajectories and sensor positions")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis("equal")
        plt.tight_layout()

        if save_file:
            filename: str = os.path.join(self.output_dir, "trajectories.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Trajectories saved to: {filename}")

        plt.show()

    def create_animation(
        self, interval: int = 100, save_gif: bool = True
    ) -> FuncAnimation:
        fig, ax = plt.subplots(figsize=(12, 10))
        time_points: np.ndarray = self.simulation_data["time_points"]

        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_title("Target movement animation")
        ax.grid(True, alpha=0.3)

        all_x: List[float] = []
        all_y: List[float] = []
        for obj in self.targets:
            positions: List[Tuple[float, float]] = [
                self.get_target_position(obj, t) for t in time_points
            ]
            all_x.extend([p[0] for p in positions])
            all_y.extend([p[1] for p in positions])
        for sensor in self.sensors:
            all_x.append(sensor.position[0])
            all_y.append(sensor.position[1])

        margin: int = 5
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        obj_points: List[Any] = []
        obj_trails: List[Any] = []
        obj_colors: List[str] = list(mcolors.TABLEAU_COLORS.keys())

        for i, obj in enumerate(self.targets):
            color: str = obj_colors[i % len(obj_colors)]
            (trail,) = ax.plot([], [], color=color, linewidth=2, alpha=0.5)
            (point,) = ax.plot(
                [], [], color=color, marker="o", markersize=10, markeredgecolor="black"
            )
            obj_trails.append(trail)
            obj_points.append(point)

            ax.text(
                obj.initial_position[0],
                obj.initial_position[1],
                f"O{obj.id}({str(obj.movement_type)})",
                fontsize=10,
                fontweight="bold",
            )

        sensor_colors: List[str] = ["red", "green", "blue", "purple", "orange", "brown"]
        for i, sensor in enumerate(self.sensors):
            color: str = sensor_colors[i % len(sensor_colors)]
            ax.scatter(
                sensor.position[0],
                sensor.position[1],
                color=color,
                s=150,
                marker="^",
                label=f"Sensor {sensor.id}",
                edgecolors="black",
                zorder=5,
            )
            ax.annotate(
                f"S{sensor.id}",
                (sensor.position[0], sensor.position[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
            )

        time_text: Any = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        def animate(frame: int) -> List[Any]:
            current_time: float = time_points[frame]
            time_text.set_text(f"Time: {current_time:.1f} sec")

            for i, obj in enumerate(self.targets):
                trail_x: List[float] = []
                trail_y: List[float] = []
                for t in time_points[: frame + 1]:
                    pos: Tuple[float, float] = self.get_target_position(obj, t)
                    trail_x.append(pos[0])
                    trail_y.append(pos[1])

                obj_trails[i].set_data(trail_x, trail_y)

                current_pos: Tuple[float, float] = self.get_target_position(
                    obj, current_time
                )
                obj_points[i].set_data([current_pos[0]], [current_pos[1]])

            return obj_trails + obj_points + [time_text]

        anim: FuncAnimation = FuncAnimation(
            fig,
            animate,
            frames=len(time_points),
            interval=interval,
            blit=True,
            repeat=True,
        )

        plt.legend()
        plt.tight_layout()

        if save_gif:
            try:
                gif_filename: str = os.path.join(self.output_dir, "animation.gif")
                anim.save(
                    gif_filename, writer=PillowWriter(fps=1000 // interval), dpi=100
                )
                print(f"Animation saved to: {gif_filename}")
            except Exception as e:
                print(f"Error saving GIF: {e}")

        plt.show()

        return anim

    def save_simulation(self, filename: str = "simulation_data.pkl") -> None:
        filepath: str = os.path.join(self.output_dir, filename)
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "duration": self.duration,
                    "time_step": self.time_step,
                    "sensors": self.sensors,
                    "targets": self.targets,
                    "simulation_data": self.simulation_data,
                    "noise_config": self.noise_config,
                },
                f,
            )
        print(f"Simulation saved to: {filepath}")

    def load_simulation(self, filename: str = "simulation_data.pkl") -> None:
        filepath: str = os.path.join(self.output_dir, filename)
        with open(filepath, "rb") as f:
            data: Dict[str, Any] = pickle.load(f)
            self.duration = data["duration"]
            self.time_step = data["time_step"]
            self.sensors = data["sensors"]
            self.targets = data["targets"]
            self.simulation_data = data["simulation_data"]
            self.noise_config = data.get("noise_config", {})
        print(f"Simulation loaded from: {filepath}")

    def print_simulation_info(self) -> None:
        print(f"Simulation duration: {self.duration} seconds")
        print(f"Time step: {self.time_step} seconds")
        print(f"Number of sensors: {len(self.sensors)}")
        print(f"Number of targets: {len(self.targets)}")
        print(f"Noise configuration: {self.noise_config}")
        print("\nSensors:")
        for sensor in self.sensors:
            print(
                f"  Sensor {sensor.id}: position ({sensor.position[0]:.1f}, {sensor.position[1]:.1f})"
            )
        print("\nTargets:")
        for obj in self.targets:
            movement_info: str = f", movement: {obj.movement_type}"
            if obj.movement_type == TARGET_TYPE.RANDOM_WALK and obj.random_walk_params:
                movement_info += f", params: {obj.random_walk_params}"
            print(
                f"  Target {obj.id}: initial position ({obj.initial_position[0]:.1f}, {obj.initial_position[1]:.1f}), "
                f"velocity ({obj.velocity[0]:.2f}, {obj.velocity[1]:.2f}){movement_info}"
            )

    def _apply_noise_to_distance(self, distance: float) -> float:
        if not self.noise_config:
            return distance

        noise_type: str = self.noise_config.get("type", "uniform")
        if noise_type == "uniform":
            low: float = self.noise_config.get("low", -0.1)
            high: float = self.noise_config.get("high", 0.1)
            noise: float = random.uniform(low, high)
            return distance + noise
        elif noise_type == "gaussian":
            mean: float = self.noise_config.get("mean", 0.0)
            std: float = self.noise_config.get("std", 0.1)
            noise: float = random.gauss(mean, std)
            return distance + noise
        else:
            return distance

    def get_spsa_input_data(
        self,
        time_points: Optional[List[float]] = None,
        initial_estimates: Optional[Dict[int, Dict[int, np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        if not hasattr(self, "simulation_data") or not self.simulation_data:
            raise ValueError(
                "Simulation must be run first. Call run_simulation() before this method."
            )

        if time_points is None:
            time_points = self.simulation_data["time_points"]

        sensors_positions: Dict[int, np.ndarray] = {
            sensor.id: np.array(sensor.position) for sensor in self.sensors
        }

        initial_estimates = {}
        for target in self.targets:
            initial_estimates[target.id] = {}
            for sensor in self.sensors:
                random_x: float = random.uniform(0, 50)
                random_y: float = random.uniform(0, 50)
                estimated_pos: np.ndarray = np.array([random_x, random_y])
                initial_estimates[target.id][sensor.id] = estimated_pos

        spsa_data: Dict[int, Any] = {}

        for i, time in enumerate(time_points):
            true_positions: Dict[int, np.ndarray] = {
                target.id: np.array(self.get_target_position_at_time(target.id, time))
                for target in self.targets
            }

            distances: Dict[int, Dict[int, float]] = {}
            for target in self.targets:
                distances[target.id] = {}
                for sensor in self.sensors:
                    distance: float = self.get_distance(sensor.id, target.id, time)
                    noisy_distance: float = self._apply_noise_to_distance(distance)
                    distances[target.id][sensor.id] = noisy_distance

            spsa_data[i] = [true_positions, distances]

        return {
            "sensors_positions": sensors_positions,
            "init_coords": initial_estimates,
            "data": spsa_data,
            "num_sensors": len(self.sensors),
        }