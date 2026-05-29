import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import random
from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpRequest, HttpResponse
from django.urls import reverse
from django.contrib import messages
from typing import Dict, List, Any, Optional, Tuple
from .simulation import Simulation
from algorithms.original_spsa import Original_SPSA
from algorithms.accelerated_spsa import Accelerated_SPSA
from algorithms.distributed_kalman_filter import Distributed_Kalman_Filter
from container_executor import algorithm_executor

matplotlib.use("Agg")


def generate_adjacency_matrix(
    num_sensors: int, sparsity_percent: float = 100.0
) -> List[List[int]]:
    if sparsity_percent >= 100.0:
        return [
            [1 if i != j else 0 for j in range(num_sensors)] for i in range(num_sensors)
        ]

    if sparsity_percent <= 0.0:
        return [[0 for j in range(num_sensors)] for i in range(num_sensors)]

    probability = sparsity_percent / 100.0
    matrix = []
    for i in range(num_sensors):
        row = []
        for j in range(num_sensors):
            if i == j:
                row.append(0)
            else:
                if random.random() < probability:
                    row.append(1)
                else:
                    row.append(0)
        matrix.append(row)

    for i in range(num_sensors):
        for j in range(num_sensors):
            if matrix[i][j] == 1:
                matrix[j][i] = 1

    return matrix


def get_algorithm_instance(algorithm_name, algorithm_config, user=None):
    base_algorithms = {
        "original_spsa": Original_SPSA,
        "accelerated_spsa": Accelerated_SPSA,
        "distributed_kalman_filter": Distributed_Kalman_Filter,
    }

    if algorithm_name in base_algorithms:
        return base_algorithms[algorithm_name](**algorithm_config)

    if user and user.is_authenticated:
        try:
            from accounts.models import CustomAlgorithm

            custom_algo = CustomAlgorithm.objects.filter(
                user=user, name=algorithm_name, is_active=True
            ).first()

            if custom_algo:
                algorithm_class = custom_algo.get_algorithm_class()
                if algorithm_class:
                    return algorithm_class(**algorithm_config)
                else:
                    print(f"Could not find algorithm class in {algorithm_name}")
                    return None
        except Exception as e:
            print(f"Error loading custom algorithm {algorithm_name}: {e}")
            return None

    return None


def run_custom_algorithm_in_container(
    algorithm_name, algorithm_config, simulation_data, user
):
    if not user or not user.is_authenticated:
        return None

    try:
        from accounts.models import CustomAlgorithm

        custom_algo = CustomAlgorithm.objects.filter(
            user=user, name=algorithm_name, is_active=True
        ).first()

        if not custom_algo:
            return None

        algorithm_class_name = custom_algo.get_algorithm_class_name()
        if not algorithm_class_name:
            return None

        result = algorithm_executor.execute_algorithm(
            custom_algo.file.path,
            algorithm_class_name,
            algorithm_config,
            simulation_data,
        )

        if result is None:
            return None

        reconstructed_result = {}
        for iteration, value in result.items():
            if isinstance(value, list) and len(value) == 2:
                true_positions = value[0]
                estimates = value[1]

                reconstructed_true = {}
                for target_id, pos in true_positions.items():
                    if isinstance(pos, list):
                        reconstructed_true[target_id] = np.array(pos)
                    else:
                        reconstructed_true[target_id] = pos

                reconstructed_estimates = {}
                for target_id, sensor_dict in estimates.items():
                    reconstructed_estimates[target_id] = {}
                    for sensor_id, pos in sensor_dict.items():
                        if isinstance(pos, list):
                            reconstructed_estimates[target_id][sensor_id] = np.array(
                                pos
                            )
                        else:
                            reconstructed_estimates[target_id][sensor_id] = pos

                reconstructed_result[iteration] = [
                    reconstructed_true,
                    reconstructed_estimates,
                ]

        return reconstructed_result

    except Exception as e:
        print(f"Error running custom algorithm {algorithm_name} in container: {e}")
        return None


def setup_view(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        duration: float = float(request.POST.get("duration", 50))
        num_sensors: int = int(request.POST.get("num_sensors", 3))
        num_linear_targets: int = int(request.POST.get("num_linear_targets", 2))
        num_random_targets: int = int(request.POST.get("num_random_targets", 2))
        algorithms: List[str] = request.POST.getlist("algorithms")
        num_runs: int = int(request.POST.get("num_runs", 1))

        noise_enabled: bool = request.POST.get("noise_enabled") == "on"
        noise_type: str = request.POST.get("noise_type", "uniform")
        noise_low: float = float(request.POST.get("noise_low", -0.1))
        noise_high: float = float(request.POST.get("noise_high", 0.1))
        noise_mean: float = float(request.POST.get("noise_mean", 0.0))
        noise_std: float = float(request.POST.get("noise_std", 0.1))

        lline_config: Dict[str, bool] = {}
        for algo in algorithms:
            if algo in [
                "original_spsa",
                "accelerated_spsa",
                "distributed_kalman_filter",
            ]:
                lline_config[algo] = request.POST.get(f"{algo}_lline") == "on"
            else:
                lline_config[algo] = request.POST.get(f"{algo}_lline") == "on"

        adjacency_sparsity: float = float(request.POST.get("adjacency_sparsity", 100))

        adjacency_matrix_str = request.POST.get("adjacency_matrix", "")
        adjacency_matrix = None
        if adjacency_matrix_str:
            try:
                rows = adjacency_matrix_str.strip().split("\n")
                adjacency_matrix = []
                for row in rows:
                    adjacency_matrix.append([int(x.strip()) for x in row.split(",")])
                if len(adjacency_matrix) != num_sensors:
                    adjacency_matrix = None
                else:
                    for row in adjacency_matrix:
                        if len(row) != num_sensors:
                            adjacency_matrix = None
                            break
            except:
                adjacency_matrix = None

        simulation_params = {
            "duration": duration,
            "num_sensors": num_sensors,
            "num_linear_targets": num_linear_targets,
            "num_random_targets": num_random_targets,
            "algorithms": algorithms,
            "noise_enabled": noise_enabled,
            "noise_type": noise_type,
            "noise_low": noise_low,
            "noise_high": noise_high,
            "noise_mean": noise_mean,
            "noise_std": noise_std,
            "num_runs": num_runs,
            "lline_config": lline_config,
            "adjacency_matrix": adjacency_matrix,
            "adjacency_sparsity": adjacency_sparsity,
        }

        request.session["simulation_params"] = simulation_params
        if "single_config_name" in request.session:
            del request.session["single_config_name"]

        return HttpResponseRedirect(reverse("simulations:results"))

    custom_algorithms = []
    if request.user.is_authenticated:
        try:
            from accounts.models import CustomAlgorithm

            custom_algorithms = list(
                request.user.algorithms.filter(is_active=True).values_list(
                    "name", flat=True
                )
            )
        except:
            pass

    return render(
        request,
        "simulations/setup.html",
        {
            "custom_algorithms": custom_algorithms,
            "user": request.user,
        },
    )


def results_view(request: HttpRequest) -> HttpResponse:
    params: Dict[str, Any] = request.session.get("simulation_params", {})

    if not params:
        return HttpResponseRedirect(reverse("simulations:setup"))

    duration: float = params.get("duration", 50)
    num_sensors: int = params.get("num_sensors", 3)
    num_linear_targets: int = params.get("num_linear_targets", 2)
    num_random_targets: int = params.get("num_random_targets", 2)
    algorithms: List[str] = params.get("algorithms", ["original_spsa"])
    noise_enabled: bool = params.get("noise_enabled", False)
    noise_type: str = params.get("noise_type", "uniform")
    noise_low: float = params.get("noise_low", -0.1)
    noise_high: float = params.get("noise_high", 0.1)
    noise_mean: float = params.get("noise_mean", 0.0)
    noise_std: float = params.get("noise_std", 0.1)
    num_runs: int = params.get("num_runs", 1)
    lline_config: Dict[str, bool] = params.get("lline_config", {})
    adjacency_matrix: Optional[List[List[int]]] = params.get("adjacency_matrix", None)
    adjacency_sparsity: float = params.get("adjacency_sparsity", 100.0)

    noise_config: Optional[Dict[str, Any]] = None
    if noise_enabled:
        if noise_type == "uniform":
            noise_config = {"type": "uniform", "low": noise_low, "high": noise_high}
        elif noise_type == "gaussian":
            noise_config = {"type": "gaussian", "mean": noise_mean, "std": noise_std}

    all_results: Dict[int, Dict[str, Any]] = {}
    all_simulations: Dict[int, Simulation] = {}
    all_initial_estimates: Dict[int, Dict[int, Dict[int, np.ndarray]]] = {}

    for run_id in range(num_runs):
        sim: Simulation = Simulation(
            duration=duration, time_step=1.0, noise_config=noise_config
        )

        for i in range(num_sensors):
            sim.add_uniform_sensor(i, area_size=50)

        target_id: int = 0
        for i in range(num_linear_targets):
            sim.add_linear_target(target_id, area_size=50)
            target_id += 1

        for i in range(num_random_targets):
            sim.add_random_walk_target(target_id, area_size=50)
            target_id += 1

        sim.run_simulation()
        spsa_input: Dict[str, Any] = sim.get_spsa_input_data()

        if adjacency_matrix is not None:
            spsa_input["adjacency_matrix"] = adjacency_matrix
        else:
            spsa_input["adjacency_matrix"] = generate_adjacency_matrix(
                num_sensors, adjacency_sparsity
            )

        all_initial_estimates[run_id] = spsa_input["init_coords"]

        results: Dict[str, Any] = {}

        for algorithm_name in algorithms:
            algorithm_config = {
                "sensors_positions": spsa_input["sensors_positions"],
                "true_targets_position": spsa_input["data"][0][0],
                "distances": spsa_input["data"][0][1],
                "init_coords": spsa_input["init_coords"],
                "adjacency_matrix": spsa_input["adjacency_matrix"],
            }

            if algorithm_name in [
                "original_spsa",
                "accelerated_spsa",
                "distributed_kalman_filter",
            ]:
                algorithm_instance = get_algorithm_instance(
                    algorithm_name,
                    algorithm_config,
                    request.user if request.user.is_authenticated else None,
                )

                if algorithm_instance:
                    try:
                        results[algorithm_name] = algorithm_instance.run_n_iterations(
                            data=spsa_input["data"]
                        )
                    except Exception as e:
                        messages.error(
                            request, f"Error running algorithm {algorithm_name}: {e}"
                        )
                else:
                    messages.warning(
                        request, f"Could not load algorithm: {algorithm_name}"
                    )
            else:
                container_result = run_custom_algorithm_in_container(
                    algorithm_name,
                    algorithm_config,
                    spsa_input["data"],
                    request.user if request.user.is_authenticated else None,
                )

                if container_result is not None:
                    results[algorithm_name] = container_result
                else:
                    algorithm_instance = get_algorithm_instance(
                        algorithm_name,
                        algorithm_config,
                        request.user if request.user.is_authenticated else None,
                    )

                    if algorithm_instance:
                        try:
                            results[algorithm_name] = (
                                algorithm_instance.run_n_iterations(
                                    data=spsa_input["data"]
                                )
                            )
                        except Exception as e:
                            messages.error(
                                request,
                                f"Error running algorithm {algorithm_name}: {e}",
                            )
                    else:
                        messages.warning(
                            request, f"Could not load algorithm: {algorithm_name}"
                        )

        all_results[run_id] = results
        all_simulations[run_id] = sim

    selected_run: Optional[int] = request.GET.get("run")
    selected_sensor: Optional[str] = request.GET.get("sensor")
    selected_target: Optional[str] = request.GET.get("target")

    if selected_run is None or selected_run == "":
        selected_run = 0
    else:
        selected_run = int(selected_run)

    selected_run = max(0, min(selected_run, num_runs - 1))

    plots_data: Dict[str, str] = {}
    aggregated_plots: Dict[str, str] = {}

    if num_runs == 1:
        plots_data = generate_plots(
            all_simulations[0],
            all_results[0],
            all_initial_estimates[0],
            selected_run,
            num_runs,
            lline_config,
            request,
        )
    else:
        aggregated_plots = generate_aggregated_plots(
            all_simulations,
            all_results,
            all_initial_estimates,
            num_runs,
            lline_config,
            request,
        )
        if selected_run < num_runs:
            plots_data = generate_plots(
                all_simulations[selected_run],
                all_results[selected_run],
                all_initial_estimates[selected_run],
                selected_run,
                num_runs,
                lline_config,
                request,
            )

    config_name = request.session.get("single_config_name", None)

    context: Dict[str, Any] = {
        "plots": plots_data,
        "aggregated_plots": aggregated_plots,
        "results": all_results.get(selected_run, {}),
        "sensors": list(range(num_sensors)),
        "targets": list(range(num_linear_targets + num_random_targets)),
        "algorithms": algorithms,
        "simulation_params": params,
        "num_runs": num_runs,
        "selected_run": selected_run,
        "run_range": range(num_runs),
        "lline_config": lline_config,
        "user": request.user,
        "config_name": config_name,
        "adjacency_sparsity": adjacency_sparsity,
    }

    if (selected_sensor is not None and selected_sensor != "") or (
        selected_target is not None and selected_target != ""
    ):
        sensor_int: Optional[int] = (
            int(selected_sensor) if selected_sensor and selected_sensor != "" else None
        )
        target_int: Optional[int] = (
            int(selected_target) if selected_target and selected_target != "" else None
        )
        individual_plots: Dict[str, str] = generate_individual_plots(
            all_simulations[selected_run],
            all_results[selected_run],
            all_initial_estimates[selected_run],
            sensor_int,
            target_int,
            selected_run,
            lline_config,
            request,
        )
        context["individual_plots"] = individual_plots
        context["selected_sensor"] = sensor_int
        context["selected_target"] = target_int

    return render(request, "simulations/results.html", context)


def comparison_results_view(request: HttpRequest) -> HttpResponse:
    multiple_configs = request.session.get("multiple_configs", [])

    if not multiple_configs:
        return HttpResponseRedirect(reverse("simulations:setup"))

    all_config_results = []

    for config_data in multiple_configs:
        config_name = config_data["name"]
        params = config_data["params"]

        duration = params.get("duration", 50)
        num_sensors = params.get("num_sensors", 3)
        num_linear_targets = params.get("num_linear_targets", 2)
        num_random_targets = params.get("num_random_targets", 2)
        algorithms = params.get("algorithms", ["original_spsa"])
        noise_enabled = params.get("noise_enabled", False)
        noise_type = params.get("noise_type", "uniform")
        noise_low = params.get("noise_low", -0.1)
        noise_high = params.get("noise_high", 0.1)
        noise_mean = params.get("noise_mean", 0.0)
        noise_std = params.get("noise_std", 0.1)
        num_runs = params.get("num_runs", 1)
        lline_config = params.get("lline_config", {})
        adjacency_matrix = params.get("adjacency_matrix", None)
        adjacency_sparsity = params.get("adjacency_sparsity", 100.0)

        noise_config = None
        if noise_enabled:
            if noise_type == "uniform":
                noise_config = {"type": "uniform", "low": noise_low, "high": noise_high}
            elif noise_type == "gaussian":
                noise_config = {
                    "type": "gaussian",
                    "mean": noise_mean,
                    "std": noise_std,
                }

        all_aggregated_errors = {algo: [] for algo in algorithms}

        for run_id in range(num_runs):
            sim = Simulation(
                duration=duration, time_step=1.0, noise_config=noise_config
            )

            for i in range(num_sensors):
                sim.add_uniform_sensor(i, area_size=50)

            target_id = 0
            for i in range(num_linear_targets):
                sim.add_linear_target(target_id, area_size=50)
                target_id += 1

            for i in range(num_random_targets):
                sim.add_random_walk_target(target_id, area_size=50)
                target_id += 1

            sim.run_simulation()
            spsa_input = sim.get_spsa_input_data()

            if adjacency_matrix is not None:
                spsa_input["adjacency_matrix"] = adjacency_matrix
            else:
                spsa_input["adjacency_matrix"] = generate_adjacency_matrix(
                    num_sensors, adjacency_sparsity
                )

            for algorithm_name in algorithms:
                algorithm_config = {
                    "sensors_positions": spsa_input["sensors_positions"],
                    "true_targets_position": spsa_input["data"][0][0],
                    "distances": spsa_input["data"][0][1],
                    "init_coords": spsa_input["init_coords"],
                    "adjacency_matrix": spsa_input["adjacency_matrix"],
                }

                if algorithm_name in [
                    "original_spsa",
                    "accelerated_spsa",
                    "distributed_kalman_filter",
                ]:
                    algorithm_instance = get_algorithm_instance(
                        algorithm_name,
                        algorithm_config,
                        request.user if request.user.is_authenticated else None,
                    )

                    if algorithm_instance:
                        try:
                            results = algorithm_instance.run_n_iterations(
                                spsa_input["data"]
                            )

                            errors_over_time = []
                            for time_iter in results.values():
                                true_positions = time_iter[0]
                                estimates = time_iter[1]
                                iteration_errors = []
                                for target_id, true_pos in true_positions.items():
                                    sensor_estimates = estimates[target_id]
                                    for sensor_est in sensor_estimates.values():
                                        error = np.linalg.norm(sensor_est - true_pos)
                                        iteration_errors.append(error)
                                if iteration_errors:
                                    errors_over_time.append(np.mean(iteration_errors))

                            if errors_over_time:
                                all_aggregated_errors[algorithm_name].append(
                                    errors_over_time
                                )
                        except Exception as e:
                            print(f"Error running {algorithm_name}: {e}")
                else:
                    container_result = run_custom_algorithm_in_container(
                        algorithm_name,
                        algorithm_config,
                        spsa_input["data"],
                        request.user if request.user.is_authenticated else None,
                    )

                    if container_result is not None:
                        errors_over_time = []
                        for time_iter in container_result.values():
                            true_positions = time_iter[0]
                            estimates = time_iter[1]
                            iteration_errors = []
                            for target_id, true_pos in true_positions.items():
                                sensor_estimates = estimates[target_id]
                                for sensor_est in sensor_estimates.values():
                                    error = np.linalg.norm(sensor_est - true_pos)
                                    iteration_errors.append(error)
                            if iteration_errors:
                                errors_over_time.append(np.mean(iteration_errors))

                        if errors_over_time:
                            all_aggregated_errors[algorithm_name].append(
                                errors_over_time
                            )
                    else:
                        algorithm_instance = get_algorithm_instance(
                            algorithm_name,
                            algorithm_config,
                            request.user if request.user.is_authenticated else None,
                        )

                        if algorithm_instance:
                            try:
                                results = algorithm_instance.run_n_iterations(
                                    spsa_input["data"]
                                )

                                errors_over_time = []
                                for time_iter in results.values():
                                    true_positions = time_iter[0]
                                    estimates = time_iter[1]
                                    iteration_errors = []
                                    for target_id, true_pos in true_positions.items():
                                        sensor_estimates = estimates[target_id]
                                        for sensor_est in sensor_estimates.values():
                                            error = np.linalg.norm(
                                                sensor_est - true_pos
                                            )
                                            iteration_errors.append(error)
                                    if iteration_errors:
                                        errors_over_time.append(
                                            np.mean(iteration_errors)
                                        )

                                if errors_over_time:
                                    all_aggregated_errors[algorithm_name].append(
                                        errors_over_time
                                    )
                            except Exception as e:
                                print(f"Error running {algorithm_name}: {e}")

        comparison_plots = generate_comparison_plots(
            all_aggregated_errors, algorithms, config_name, lline_config, request
        )

        all_config_results.append(
            {
                "name": config_name,
                "params": params,
                "plots": comparison_plots,
                "algorithms": algorithms,
            }
        )

    if "multiple_configs" in request.session:
        del request.session["multiple_configs"]
    if "single_config_name" in request.session:
        del request.session["single_config_name"]

    return render(
        request,
        "simulations/comparison_results.html",
        {
            "config_results": all_config_results,
            "user": request.user,
        },
    )


def generate_plots(
    sim: Simulation,
    results: Dict[str, Any],
    init_coords: Dict[int, Dict[int, np.ndarray]],
    run_id: int,
    num_runs: int,
    lline_config: Dict[str, bool],
    request: HttpRequest = None,
) -> Dict[str, str]:
    plots: Dict[str, str] = {}

    is_russian = False
    if request:
        language = request.LANGUAGE_CODE
        is_russian = language == "ru"

    plt.figure(figsize=(12, 8))

    colors: np.ndarray = plt.cm.tab10(np.linspace(0, 1, len(sim.targets)))

    for i, target in enumerate(sim.targets):
        positions: List[Any] = [
            sim.get_target_position(target, t)
            for t in sim.simulation_data["time_points"]
        ]
        x_vals: List[float] = [p[0] for p in positions]
        y_vals: List[float] = [p[1] for p in positions]

        label_true = f"Target {target.id} (True)"
        if is_russian:
            label_true = f"Цель {target.id} (Истинная)"

        plt.plot(
            x_vals,
            y_vals,
            color=colors[i],
            linewidth=3,
            label=label_true,
            alpha=0.7,
        )

        plt.scatter(
            x_vals[0],
            y_vals[0],
            color=colors[i],
            s=180,
            marker="D",
            edgecolors="black",
            linewidth=2,
            zorder=5,
        )

        plt.scatter(
            x_vals[-1],
            y_vals[-1],
            color=colors[i],
            s=180,
            marker="X",
            edgecolors="black",
            linewidth=2,
            zorder=5,
        )

    line_styles = {
        "original_spsa": "-",
        "accelerated_spsa": ":",
        "distributed_kalman_filter": "--",
    }
    for algorithm_name, algorithm_results in results.items():
        if algorithm_name not in line_styles:
            line_styles[algorithm_name] = "--"

        for target_id in algorithm_results[0][0].keys():
            target_estimates: List[np.ndarray] = []

            initial_est = init_coords[target_id][0]
            target_estimates.append(initial_est)

            for time_iter in algorithm_results.values():
                estimates_at_time: Dict[int, np.ndarray] = time_iter[1][target_id]
                avg_estimate: np.ndarray = np.mean(
                    list(estimates_at_time.values()), axis=0
                )
                target_estimates.append(avg_estimate)

            x_vals: List[float] = [est[0] for est in target_estimates]
            y_vals: List[float] = [est[1] for est in target_estimates]

            label_est = f"Target {target_id} ({algorithm_name})"
            if is_russian:
                label_est = f"Цель {target_id} ({algorithm_name})"

            plt.plot(
                x_vals,
                y_vals,
                line_styles.get(algorithm_name, "--"),
                color=colors[target_id],
                linewidth=2,
                label=label_est,
                alpha=0.8,
            )

            plt.scatter(
                x_vals[0],
                y_vals[0],
                color=colors[target_id],
                s=120,
                marker="s",
                edgecolors="black",
                linewidth=2,
                zorder=5,
            )

            plt.scatter(
                x_vals[-1],
                y_vals[-1],
                color=colors[target_id],
                s=120,
                marker="o",
                edgecolors="black",
                linewidth=2,
                zorder=5,
            )

    for i, sensor in enumerate(sim.sensors):
        label_sensor = f"Sensor {sensor.id}"
        if is_russian:
            label_sensor = f"Сенсор {sensor.id}"

        plt.scatter(
            sensor.position[0],
            sensor.position[1],
            color="red",
            s=150,
            marker="^",
            label=label_sensor,
            edgecolors="black",
            zorder=5,
        )
        plt.annotate(
            f"S{i}",
            (sensor.position[0], sensor.position[1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontweight="bold",
        )

    label_start_true = "Start (True)"
    label_end_true = "End (True)"
    label_start_est = "Start (Est.)"
    label_end_est = "End (Est.)"

    if is_russian:
        label_start_true = "Начало (Истинное)"
        label_end_true = "Конец (Истинный)"
        label_start_est = "Начало (Оценка)"
        label_end_est = "Конец (Оценка)"

    plt.scatter(
        [],
        [],
        c="white",
        s=180,
        marker="D",
        edgecolors="black",
        linewidth=2,
        label=label_start_true,
    )
    plt.scatter(
        [],
        [],
        c="white",
        s=180,
        marker="X",
        edgecolors="black",
        linewidth=2,
        label=label_end_true,
    )
    plt.scatter(
        [],
        [],
        c="white",
        s=120,
        marker="s",
        edgecolors="black",
        linewidth=2,
        label=label_start_est,
    )
    plt.scatter(
        [],
        [],
        c="white",
        s=120,
        marker="o",
        edgecolors="black",
        linewidth=2,
        label=label_end_est,
    )

    xlabel = "X coordinate"
    ylabel = "Y coordinate"
    title_trajectories = "True Trajectories and Algorithm Estimates"
    title_convergence = "Convergence Error for Each Target"

    if is_russian:
        xlabel = "Координата X"
        ylabel = "Координата Y"
        if num_runs > 1:
            title_trajectories = f"Истинные траектории и оценки алгоритмов (Запуск {run_id + 1}/{num_runs})"
            title_convergence = (
                f"Ошибка сходимости для каждой цели (Запуск {run_id + 1}/{num_runs})"
            )
        else:
            title_trajectories = "Истинные траектории и оценки алгоритмов"
            title_convergence = "Ошибка сходимости для каждой цели"
    else:
        if num_runs > 1:
            title_trajectories = f"True Trajectories and Algorithm Estimates (Run {run_id + 1}/{num_runs})"
            title_convergence = (
                f"Convergence Error for Each Target (Run {run_id + 1}/{num_runs})"
            )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title_trajectories)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    buffer: io.BytesIO = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    plots["trajectories"] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    plt.figure(figsize=(12, 8))

    line_styles = {
        "original_spsa": "-",
        "accelerated_spsa": ":",
        "distributed_kalman_filter": "--",
    }
    for algorithm_name, algorithm_results in results.items():
        if algorithm_name not in line_styles:
            line_styles[algorithm_name] = "--"

        errors_over_time: Dict[int, List[float]] = {
            target_id: [] for target_id in algorithm_results[0][0].keys()
        }

        for target_id in algorithm_results[0][0].keys():
            initial_est = init_coords[target_id][0]
            true_pos = algorithm_results[0][0][target_id]
            initial_error = np.linalg.norm(initial_est - true_pos)
            errors_over_time[target_id].append(initial_error)

        for time_iter in algorithm_results.values():
            true_positions: Dict[int, np.ndarray] = time_iter[0]
            estimates: Dict[int, Dict[int, np.ndarray]] = time_iter[1]

            for target_id, true_pos in true_positions.items():
                sensor_estimates: Dict[int, np.ndarray] = estimates[target_id]
                errors: List[float] = []
                for sensor_est in sensor_estimates.values():
                    error: float = np.linalg.norm(sensor_est - true_pos)
                    errors.append(error)
                avg_error: float = np.mean(errors)
                errors_over_time[target_id].append(avg_error)

        for target_id, errors in errors_over_time.items():
            label = f"Target {target_id} ({algorithm_name})"
            if is_russian:
                label = f"Цель {target_id} ({algorithm_name})"

            plt.plot(
                range(len(errors)),
                errors,
                color=colors[target_id],
                linestyle=line_styles.get(algorithm_name, "-"),
                label=label,
                linewidth=2,
            )

    xlabel_iter = "Iteration (including initial)"
    ylabel_error = "Average Error (All Sensors)"

    if is_russian:
        xlabel_iter = "Итерация (включая начальную)"
        ylabel_error = "Средняя ошибка (все сенсоры)"

    plt.xlabel(xlabel_iter)
    plt.ylabel(ylabel_error)
    plt.title(title_convergence)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    plots["convergence"] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return plots


def generate_aggregated_plots(
    all_simulations: Dict[int, Simulation],
    all_results: Dict[int, Dict[str, Any]],
    all_initial_estimates: Dict[int, Dict[int, Dict[int, np.ndarray]]],
    num_runs: int,
    lline_config: Dict[str, bool],
    request: HttpRequest = None,
) -> Dict[str, str]:
    plots: Dict[str, str] = {}

    if num_runs <= 1:
        return plots

    is_russian = False
    if request:
        language = request.LANGUAGE_CODE
        is_russian = language == "ru"

    aggregated_errors: Dict[str, List[List[float]]] = {}

    for algorithm_name in all_results[0].keys():
        aggregated_errors[algorithm_name] = []
        for run_id in range(num_runs):
            if run_id in all_results and algorithm_name in all_results[run_id]:
                algorithm_results = all_results[run_id][algorithm_name]
                errors_over_time: List[float] = []

                init_coords = all_initial_estimates[run_id]

                initial_iteration_errors: List[float] = []
                for target_id, true_pos in algorithm_results[0][0].items():
                    initial_est = init_coords[target_id][0]
                    error: float = np.linalg.norm(initial_est - true_pos)
                    initial_iteration_errors.append(error)
                initial_avg_error = (
                    np.mean(initial_iteration_errors)
                    if initial_iteration_errors
                    else 0.0
                )
                errors_over_time.append(initial_avg_error)

                for time_iter in algorithm_results.values():
                    true_positions: Dict[int, np.ndarray] = time_iter[0]
                    estimates: Dict[int, Dict[int, np.ndarray]] = time_iter[1]

                    iteration_errors: List[float] = []
                    for target_id, true_pos in true_positions.items():
                        sensor_estimates: Dict[int, np.ndarray] = estimates[target_id]
                        for sensor_est in sensor_estimates.values():
                            error: float = np.linalg.norm(sensor_est - true_pos)
                            iteration_errors.append(error)

                    avg_error: float = (
                        np.mean(iteration_errors) if iteration_errors else 0.0
                    )
                    errors_over_time.append(avg_error)

                aggregated_errors[algorithm_name].append(errors_over_time)

    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(aggregated_errors.keys())))

    for idx, (algorithm_name, all_run_errors) in enumerate(aggregated_errors.items()):
        if not all_run_errors:
            continue

        min_length = min(len(errors) for errors in all_run_errors)
        all_run_errors = [errors[:min_length] for errors in all_run_errors]

        mean_errors = np.mean(all_run_errors, axis=0)
        std_errors = np.std(all_run_errors, axis=0)

        iterations = range(len(mean_errors))

        plt.plot(
            iterations,
            mean_errors,
            color=colors[idx],
            label=algorithm_name,
            linewidth=2,
        )
        plt.fill_between(
            iterations,
            mean_errors - std_errors,
            mean_errors + std_errors,
            color=colors[idx],
            alpha=0.2,
        )

        if lline_config.get(algorithm_name, False):
            final_mean_error = mean_errors[-1]
            plt.axhline(
                y=final_mean_error,
                color=colors[idx],
                linestyle=":",
                alpha=0.7,
                linewidth=2,
            )
            plt.text(
                iterations[-1],
                final_mean_error * 1.05,
                f"L={final_mean_error:.3f}",
                color=colors[idx],
                fontsize=11,
                ha="right",
                fontweight="bold",
            )

    xlabel_iter = "Iteration (including initial)"
    ylabel_error = "Aggregated Error (Mean ± Std)"
    title = f"Aggregated Error Convergence ({num_runs} Runs)"

    if is_russian:
        xlabel_iter = "Итерация (включая начальную)"
        ylabel_error = "Агрегированная ошибка (Среднее ± Ст. откл.)"
        title = f"Агрегированная сходимость ошибки ({num_runs} запусков)"

    plt.xlabel(xlabel_iter)
    plt.ylabel(ylabel_error)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    plots["aggregated"] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return plots


def generate_individual_plots(
    sim: Simulation,
    results: Dict[str, Any],
    init_coords: Dict[int, Dict[int, np.ndarray]],
    sensor_id: Optional[int] = None,
    target_id: Optional[int] = None,
    run_id: int = 0,
    lline_config: Dict[str, bool] = None,
    request: HttpRequest = None,
) -> Dict[str, str]:
    if lline_config is None:
        lline_config = {}

    plots: Dict[str, str] = {}

    is_russian = False
    if request:
        language = request.LANGUAGE_CODE
        is_russian = language == "ru"

    plt.figure(figsize=(12, 8))

    if target_id is not None:
        colors: np.ndarray = plt.cm.tab10(np.linspace(0, 1, len(sim.sensors)))

        target_obj = next((t for t in sim.targets if t.id == target_id), None)
        if target_obj:
            positions: List[Any] = [
                sim.get_target_position(target_obj, t)
                for t in sim.simulation_data["time_points"]
            ]
            x_vals: List[float] = [p[0] for p in positions]
            y_vals: List[float] = [p[1] for p in positions]

            label_true = f"Target {target_id} (True)"
            if is_russian:
                label_true = f"Цель {target_id} (Истинная)"

            plt.plot(
                x_vals,
                y_vals,
                color="black",
                linewidth=4,
                label=label_true,
                alpha=0.7,
            )

            plt.scatter(
                x_vals[0],
                y_vals[0],
                color="black",
                s=200,
                marker="D",
                edgecolors="white",
                linewidth=2,
                zorder=5,
            )

            plt.scatter(
                x_vals[-1],
                y_vals[-1],
                color="black",
                s=200,
                marker="X",
                edgecolors="white",
                linewidth=2,
                zorder=5,
            )

        line_styles = {
            "original_spsa": "-",
            "accelerated_spsa": ":",
            "distributed_kalman_filter": "--",
        }
        for algorithm_name, algorithm_results in results.items():
            if algorithm_name not in line_styles:
                line_styles[algorithm_name] = "--"

            if sensor_id is not None:
                sensor_estimates: List[np.ndarray] = []

                initial_est = init_coords[target_id][sensor_id]
                sensor_estimates.append(initial_est)

                for time_iter in algorithm_results.values():
                    estimates_at_time: Dict[int, np.ndarray] = time_iter[1][target_id]
                    if sensor_id in estimates_at_time:
                        sensor_estimates.append(estimates_at_time[sensor_id])

                if sensor_estimates:
                    x_vals: List[float] = [est[0] for est in sensor_estimates]
                    y_vals: List[float] = [est[1] for est in sensor_estimates]

                    label_sensor = f"Sensor {sensor_id} ({algorithm_name})"
                    if is_russian:
                        label_sensor = f"Сенсор {sensor_id} ({algorithm_name})"

                    plt.plot(
                        x_vals,
                        y_vals,
                        line_styles.get(algorithm_name, "--"),
                        color=colors[sensor_id % len(colors)],
                        linewidth=2,
                        label=label_sensor,
                        alpha=0.8,
                    )

                    plt.scatter(
                        x_vals[0],
                        y_vals[0],
                        color=colors[sensor_id % len(colors)],
                        s=120,
                        marker="s",
                        edgecolors="black",
                        linewidth=2,
                        zorder=5,
                    )

                    plt.scatter(
                        x_vals[-1],
                        y_vals[-1],
                        color=colors[sensor_id % len(colors)],
                        s=120,
                        marker="o",
                        edgecolors="black",
                        linewidth=2,
                        zorder=5,
                    )
            else:
                for sensor_idx in range(len(sim.sensors)):
                    sensor_estimates: List[np.ndarray] = []

                    initial_est = init_coords[target_id][sensor_idx]
                    sensor_estimates.append(initial_est)

                    for time_iter in algorithm_results.values():
                        estimates_at_time: Dict[int, np.ndarray] = time_iter[1][
                            target_id
                        ]
                        if sensor_idx in estimates_at_time:
                            sensor_estimates.append(estimates_at_time[sensor_idx])

                    if sensor_estimates:
                        x_vals: List[float] = [est[0] for est in sensor_estimates]
                        y_vals: List[float] = [est[1] for est in sensor_estimates]

                        label_sensor = f"Sensor {sensor_idx} ({algorithm_name})"
                        if is_russian:
                            label_sensor = f"Сенсор {sensor_idx} ({algorithm_name})"

                        plt.plot(
                            x_vals,
                            y_vals,
                            line_styles.get(algorithm_name, "--"),
                            color=colors[sensor_idx],
                            linewidth=2,
                            label=label_sensor,
                            alpha=0.8,
                        )

                        plt.scatter(
                            x_vals[0],
                            y_vals[0],
                            color=colors[sensor_idx],
                            s=120,
                            marker="s",
                            edgecolors="black",
                            linewidth=2,
                            zorder=5,
                        )

                        plt.scatter(
                            x_vals[-1],
                            y_vals[-1],
                            color=colors[sensor_idx],
                            s=120,
                            marker="o",
                            edgecolors="black",
                            linewidth=2,
                            zorder=5,
                        )

        for i, sensor in enumerate(sim.sensors):
            if sensor_id is None or sensor.id == sensor_id:
                label_sensor_point = f"Sensor {sensor.id}"
                if is_russian:
                    label_sensor_point = f"Сенсор {sensor.id}"

                plt.scatter(
                    sensor.position[0],
                    sensor.position[1],
                    color=colors[i],
                    s=150,
                    marker="^",
                    label=label_sensor_point,
                    edgecolors="black",
                    zorder=5,
                )
                plt.annotate(
                    f"S{sensor.id}",
                    (sensor.position[0], sensor.position[1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontweight="bold",
                )

        label_start_true = "Start (True)"
        label_end_true = "End (True)"
        label_start_est = "Start (Est.)"
        label_end_est = "End (Est.)"

        if is_russian:
            label_start_true = "Начало (Истинное)"
            label_end_true = "Конец (Истинный)"
            label_start_est = "Начало (Оценка)"
            label_end_est = "Конец (Оценка)"

        plt.scatter(
            [],
            [],
            c="white",
            s=180,
            marker="D",
            edgecolors="black",
            linewidth=2,
            label=label_start_true,
        )
        plt.scatter(
            [],
            [],
            c="white",
            s=180,
            marker="X",
            edgecolors="black",
            linewidth=2,
            label=label_end_true,
        )
        plt.scatter(
            [],
            [],
            c="white",
            s=120,
            marker="s",
            edgecolors="black",
            linewidth=2,
            label=label_start_est,
        )
        plt.scatter(
            [],
            [],
            c="white",
            s=120,
            marker="o",
            edgecolors="black",
            linewidth=2,
            label=label_end_est,
        )

        title_suffix = ""
        if sensor_id is not None and target_id is not None:
            if is_russian:
                title_suffix = f" - Сенсор {sensor_id} и Цель {target_id}"
            else:
                title_suffix = f" - Sensor {sensor_id} & Target {target_id}"
        elif target_id is not None:
            if is_russian:
                title_suffix = f" - Цель {target_id}"
            else:
                title_suffix = f" - Target {target_id}"

        if is_russian:
            title_suffix += f" (Запуск {run_id + 1})"
        else:
            title_suffix += f" (Run {run_id + 1})"

        xlabel = "X coordinate"
        ylabel = "Y coordinate"
        title = f"Trajectories{title_suffix}"

        if is_russian:
            xlabel = "Координата X"
            ylabel = "Координата Y"
            title = f"Траектории{title_suffix}"

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
    else:
        colors: np.ndarray = plt.cm.tab10(np.linspace(0, 1, len(sim.targets)))

        for i, target in enumerate(sim.targets):
            positions: List[Any] = [
                sim.get_target_position(target, t)
                for t in sim.simulation_data["time_points"]
            ]
            x_vals: List[float] = [p[0] for p in positions]
            y_vals: List[float] = [p[1] for p in positions]

            label_true = f"Target {target.id} (True)"
            if is_russian:
                label_true = f"Цель {target.id} (Истинная)"

            plt.plot(
                x_vals,
                y_vals,
                color=colors[i],
                linewidth=3,
                label=label_true,
                alpha=0.7,
            )

            plt.scatter(
                x_vals[0],
                y_vals[0],
                color=colors[i],
                s=180,
                marker="D",
                edgecolors="black",
                linewidth=2,
                zorder=5,
            )

            plt.scatter(
                x_vals[-1],
                y_vals[-1],
                color=colors[i],
                s=180,
                marker="X",
                edgecolors="black",
                linewidth=2,
                zorder=5,
            )

        line_styles = {
            "original_spsa": "-",
            "accelerated_spsa": ":",
            "distributed_kalman_filter": "--",
        }
        for algorithm_name, algorithm_results in results.items():
            if algorithm_name not in line_styles:
                line_styles[algorithm_name] = "--"

            if sensor_id is not None:
                for target_idx in algorithm_results[0][0].keys():
                    sensor_estimates: List[np.ndarray] = []

                    initial_est = init_coords[target_idx][sensor_id]
                    sensor_estimates.append(initial_est)

                    for time_iter in algorithm_results.values():
                        estimates_at_time: Dict[int, np.ndarray] = time_iter[1][
                            target_idx
                        ]
                        if sensor_id in estimates_at_time:
                            sensor_estimates.append(estimates_at_time[sensor_id])

                    if sensor_estimates:
                        x_vals: List[float] = [est[0] for est in sensor_estimates]
                        y_vals: List[float] = [est[1] for est in sensor_estimates]

                        label_est = (
                            f"Target {target_idx} (Sensor {sensor_id} {algorithm_name})"
                        )
                        if is_russian:
                            label_est = f"Цель {target_idx} (Сенсор {sensor_id} {algorithm_name})"

                        plt.plot(
                            x_vals,
                            y_vals,
                            line_styles.get(algorithm_name, "--"),
                            color=colors[target_idx],
                            linewidth=2,
                            label=label_est,
                            alpha=0.8,
                        )

                        plt.scatter(
                            x_vals[0],
                            y_vals[0],
                            color=colors[target_idx],
                            s=120,
                            marker="s",
                            edgecolors="black",
                            linewidth=2,
                            zorder=5,
                        )

                        plt.scatter(
                            x_vals[-1],
                            y_vals[-1],
                            color=colors[target_idx],
                            s=120,
                            marker="o",
                            edgecolors="black",
                            linewidth=2,
                            zorder=5,
                        )

        for i, sensor in enumerate(sim.sensors):
            if sensor_id is None or sensor.id == sensor_id:
                label_sensor_point = f"Sensor {sensor.id}"
                if is_russian:
                    label_sensor_point = f"Сенсор {sensor.id}"

                plt.scatter(
                    sensor.position[0],
                    sensor.position[1],
                    color="red",
                    s=150,
                    marker="^",
                    label=label_sensor_point,
                    edgecolors="black",
                    zorder=5,
                )
                plt.annotate(
                    f"S{sensor.id}",
                    (sensor.position[0], sensor.position[1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontweight="bold",
                )

        label_start_true = "Start (True)"
        label_end_true = "End (True)"
        label_start_est = "Start (Est.)"
        label_end_est = "End (Est.)"

        if is_russian:
            label_start_true = "Начало (Истинное)"
            label_end_true = "Конец (Истинный)"
            label_start_est = "Начало (Оценка)"
            label_end_est = "Конец (Оценка)"

        plt.scatter(
            [],
            [],
            c="white",
            s=180,
            marker="D",
            edgecolors="black",
            linewidth=2,
            label=label_start_true,
        )
        plt.scatter(
            [],
            [],
            c="white",
            s=180,
            marker="X",
            edgecolors="black",
            linewidth=2,
            label=label_end_true,
        )
        plt.scatter(
            [],
            [],
            c="white",
            s=120,
            marker="s",
            edgecolors="black",
            linewidth=2,
            label=label_start_est,
        )
        plt.scatter(
            [],
            [],
            c="white",
            s=120,
            marker="o",
            edgecolors="black",
            linewidth=2,
            label=label_end_est,
        )

        title_suffix = ""
        if sensor_id is not None:
            if is_russian:
                title_suffix = f" - Сенсор {sensor_id}"
            else:
                title_suffix = f" - Sensor {sensor_id}"

        if is_russian:
            title_suffix += f" (Запуск {run_id + 1})"
        else:
            title_suffix += f" (Run {run_id + 1})"

        xlabel = "X coordinate"
        ylabel = "Y coordinate"
        title = f"Trajectories{title_suffix}"

        if is_russian:
            xlabel = "Координата X"
            ylabel = "Координата Y"
            title = f"Траектории{title_suffix}"

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

    buffer: io.BytesIO = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    plots["individual_trajectories"] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    plt.figure(figsize=(12, 8))

    line_styles = {
        "original_spsa": "-",
        "accelerated_spsa": ":",
        "distributed_kalman_filter": "--",
    }

    for algorithm_name, algorithm_results in results.items():
        if algorithm_name not in line_styles:
            line_styles[algorithm_name] = "--"

        if target_id is not None:
            if sensor_id is not None:
                errors: List[float] = []

                initial_est = init_coords[target_id][sensor_id]
                true_pos = algorithm_results[0][0][target_id]
                initial_error = np.linalg.norm(initial_est - true_pos)
                errors.append(initial_error)

                for time_iter in algorithm_results.values():
                    true_positions: Dict[int, np.ndarray] = time_iter[0]
                    estimates: Dict[int, Dict[int, np.ndarray]] = time_iter[1]

                    if target_id in true_positions and target_id in estimates:
                        if sensor_id in estimates[target_id]:
                            true_pos: np.ndarray = true_positions[target_id]
                            sensor_est: np.ndarray = estimates[target_id][sensor_id]
                            error: float = np.linalg.norm(sensor_est - true_pos)
                            errors.append(error)

                if errors:
                    label = f"Sensor {sensor_id} ({algorithm_name})"
                    if is_russian:
                        label = f"Сенсор {sensor_id} ({algorithm_name})"

                    plt.plot(
                        range(len(errors)),
                        errors,
                        linestyle=line_styles.get(algorithm_name, "-"),
                        label=label,
                        linewidth=2,
                    )

                    if lline_config.get(algorithm_name, False):
                        final_error = errors[-1]
                        plt.axhline(
                            y=final_error,
                            color="gray",
                            linestyle=":",
                            alpha=0.7,
                            linewidth=2,
                        )
                        plt.text(
                            len(errors) - 1,
                            final_error * 1.05,
                            f"L={final_error:.3f}",
                            color="gray",
                            fontsize=10,
                            ha="right",
                            fontweight="bold",
                        )
            else:
                colors_sensor: np.ndarray = plt.cm.tab10(
                    np.linspace(0, 1, len(sim.sensors))
                )
                for sensor_idx in range(len(sim.sensors)):
                    errors: List[float] = []

                    initial_est = init_coords[target_id][sensor_idx]
                    true_pos = algorithm_results[0][0][target_id]
                    initial_error = np.linalg.norm(initial_est - true_pos)
                    errors.append(initial_error)

                    for time_iter in algorithm_results.values():
                        true_positions: Dict[int, np.ndarray] = time_iter[0]
                        estimates: Dict[int, Dict[int, np.ndarray]] = time_iter[1]

                        if target_id in true_positions and target_id in estimates:
                            if sensor_idx in estimates[target_id]:
                                true_pos: np.ndarray = true_positions[target_id]
                                sensor_est: np.ndarray = estimates[target_id][
                                    sensor_idx
                                ]
                                error: float = np.linalg.norm(sensor_est - true_pos)
                                errors.append(error)

                    if errors:
                        label = f"Sensor {sensor_idx} ({algorithm_name})"
                        if is_russian:
                            label = f"Сенсор {sensor_idx} ({algorithm_name})"

                        plt.plot(
                            range(len(errors)),
                            errors,
                            color=colors_sensor[sensor_idx],
                            linestyle=line_styles.get(algorithm_name, "-"),
                            label=label,
                            linewidth=2,
                        )

                        if lline_config.get(algorithm_name, False):
                            final_error = errors[-1]
                            plt.axhline(
                                y=final_error,
                                color=colors_sensor[sensor_idx],
                                linestyle=":",
                                alpha=0.7,
                                linewidth=2,
                            )
                            plt.text(
                                len(errors) - 1,
                                final_error * 1.05,
                                f"L={final_error:.3f}",
                                color=colors_sensor[sensor_idx],
                                fontsize=10,
                                ha="right",
                                fontweight="bold",
                            )
        else:
            if sensor_id is not None:
                colors_target: np.ndarray = plt.cm.tab10(
                    np.linspace(0, 1, len(sim.targets))
                )
                for target_idx in algorithm_results[0][0].keys():
                    errors: List[float] = []

                    initial_est = init_coords[target_idx][sensor_id]
                    true_pos = algorithm_results[0][0][target_idx]
                    initial_error = np.linalg.norm(initial_est - true_pos)
                    errors.append(initial_error)

                    for time_iter in algorithm_results.values():
                        true_positions: Dict[int, np.ndarray] = time_iter[0]
                        estimates: Dict[int, Dict[int, np.ndarray]] = time_iter[1]

                        if target_idx in true_positions and target_idx in estimates:
                            if sensor_id in estimates[target_idx]:
                                true_pos: np.ndarray = true_positions[target_idx]
                                sensor_est: np.ndarray = estimates[target_idx][
                                    sensor_id
                                ]
                                error: float = np.linalg.norm(sensor_est - true_pos)
                                errors.append(error)

                    if errors:
                        label = f"Target {target_idx} ({algorithm_name})"
                        if is_russian:
                            label = f"Цель {target_idx} ({algorithm_name})"

                        plt.plot(
                            range(len(errors)),
                            errors,
                            color=colors_target[target_idx],
                            linestyle=line_styles.get(algorithm_name, "-"),
                            label=label,
                            linewidth=2,
                        )

                        if lline_config.get(algorithm_name, False):
                            final_error = errors[-1]
                            plt.axhline(
                                y=final_error,
                                color=colors_target[target_idx],
                                linestyle=":",
                                alpha=0.7,
                                linewidth=2,
                            )
                            plt.text(
                                len(errors) - 1,
                                final_error * 1.05,
                                f"L={final_error:.3f}",
                                color=colors_target[target_idx],
                                fontsize=10,
                                ha="right",
                                fontweight="bold",
                            )

    title_suffix = ""
    if sensor_id is not None and target_id is not None:
        if is_russian:
            title_suffix = f" - Сенсор {sensor_id} и Цель {target_id}"
        else:
            title_suffix = f" - Sensor {sensor_id} & Target {target_id}"
    elif sensor_id is not None:
        if is_russian:
            title_suffix = f" - Сенсор {sensor_id}"
        else:
            title_suffix = f" - Sensor {sensor_id}"
    elif target_id is not None:
        if is_russian:
            title_suffix = f" - Цель {target_id}"
        else:
            title_suffix = f" - Target {target_id}"

    if is_russian:
        title_suffix += f" (Запуск {run_id + 1})"
    else:
        title_suffix += f" (Run {run_id + 1})"

    xlabel_iter = "Iteration (including initial)"
    ylabel_error = "Error"
    title = f"Convergence Error{title_suffix}"

    if is_russian:
        xlabel_iter = "Итерация (включая начальную)"
        ylabel_error = "Ошибка"
        title = f"Ошибка сходимости{title_suffix}"

    plt.xlabel(xlabel_iter)
    plt.ylabel(ylabel_error)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    plots["individual_convergence"] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return plots


def generate_comparison_plots(
    aggregated_errors: Dict[str, List[List[float]]],
    algorithms: List[str],
    config_name: str,
    lline_config: Dict[str, bool],
    request: HttpRequest = None,
) -> Dict[str, str]:
    plots = {}

    is_russian = False
    if request:
        language = request.LANGUAGE_CODE
        is_russian = language == "ru"

    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))

    for idx, algorithm_name in enumerate(algorithms):
        if (
            algorithm_name not in aggregated_errors
            or not aggregated_errors[algorithm_name]
        ):
            continue

        all_run_errors = aggregated_errors[algorithm_name]
        min_length = min(len(errors) for errors in all_run_errors)
        all_run_errors = [errors[:min_length] for errors in all_run_errors]

        mean_errors = np.mean(all_run_errors, axis=0)
        std_errors = np.std(all_run_errors, axis=0)

        iterations = range(len(mean_errors))

        plt.plot(
            iterations,
            mean_errors,
            color=colors[idx],
            label=algorithm_name,
            linewidth=2,
        )
        plt.fill_between(
            iterations,
            mean_errors - std_errors,
            mean_errors + std_errors,
            color=colors[idx],
            alpha=0.2,
        )

        if lline_config.get(algorithm_name, False):
            final_mean_error = mean_errors[-1]
            plt.axhline(
                y=final_mean_error,
                color=colors[idx],
                linestyle=":",
                alpha=0.7,
                linewidth=2,
            )
            plt.text(
                iterations[-1],
                final_mean_error * 1.05,
                f"L={final_mean_error:.3f}",
                color=colors[idx],
                fontsize=10,
                ha="right",
                fontweight="bold",
            )

    xlabel_iter = "Iteration (including initial)"
    ylabel_error = "Average Error (Mean ± Std)"
    title = f"Convergence Comparison - {config_name}"

    if is_russian:
        xlabel_iter = "Итерация (включая начальную)"
        ylabel_error = "Средняя ошибка (Среднее ± Ст. откл.)"
        title = f"Сравнение сходимости - {config_name}"

    plt.xlabel(xlabel_iter)
    plt.ylabel(ylabel_error)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    plots["comparison"] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return plots
