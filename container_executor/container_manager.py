import docker
import tempfile
import os
import pickle
import uuid
import numpy as np
from typing import Dict, Any, Optional
from django.conf import settings


class ContainerManager:
    def __init__(self):
        self.client = None
        self.image_name = "algorithm-runner:latest"
        self._connect()

    def _connect(self):
        try:
            self.client = docker.from_env()
        except Exception:
            self.client = None

    def is_available(self) -> bool:
        return self.client is not None and settings.CONTAINER_EXECUTION_ENABLED

    def ensure_image(self) -> bool:
        if not self.is_available():
            return False

        try:
            self.client.images.get(self.image_name)
            return True
        except docker.errors.ImageNotFound:
            return self._build_image()

    def _build_image(self) -> bool:
        if not self.client:
            return False

        dockerfile_content = """
FROM python:3.12-slim

RUN pip install --no-cache-dir numpy

WORKDIR /app

COPY runner.py /app/runner.py

RUN chmod +x /app/runner.py

USER 1000:1000

ENTRYPOINT ["python", "/app/runner.py"]
"""

        runner_content = """
import sys
import json
import pickle
import numpy as np
import importlib.util
import traceback
from typing import Dict, Any


def load_algorithm(algorithm_path: str, algorithm_class_name: str):
    spec = importlib.util.spec_from_file_location("user_algorithm", algorithm_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, obj in module.__dict__.items():
        if name == algorithm_class_name:
            return obj

    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and hasattr(obj, "run_n_iterations"):
            return obj

    raise ValueError(f"Algorithm class {algorithm_class_name} not found")


def main():
    if len(sys.argv) != 2:
        print("Usage: python runner.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    with open(input_file, "rb") as f:
        data = pickle.load(f)

    algorithm_path = data["algorithm_path"]
    algorithm_class_name = data["algorithm_class_name"]
    algorithm_config = data["algorithm_config"]
    simulation_data = data["simulation_data"]

    try:
        algorithm_class = load_algorithm(algorithm_path, algorithm_class_name)
        algorithm_instance = algorithm_class(**algorithm_config)

        result = algorithm_instance.run_n_iterations(simulation_data)

        serializable_result = {}
        for iteration, value in result.items():
            if isinstance(value, list) and len(value) == 2:
                true_positions = value[0]
                estimates = value[1]

                serializable_true = {}
                for target_id, pos in true_positions.items():
                    if hasattr(pos, "tolist"):
                        serializable_true[target_id] = pos.tolist()
                    else:
                        serializable_true[target_id] = pos

                serializable_estimates = {}
                for target_id, sensor_dict in estimates.items():
                    serializable_estimates[target_id] = {}
                    for sensor_id, pos in sensor_dict.items():
                        if hasattr(pos, "tolist"):
                            serializable_estimates[target_id][sensor_id] = pos.tolist()
                        else:
                            serializable_estimates[target_id][sensor_id] = pos

                serializable_result[iteration] = [serializable_true, serializable_estimates]

        with open(input_file + ".out", "wb") as f:
            pickle.dump(serializable_result, f)

        print("SUCCESS")

    except Exception as e:
        error_msg = f"ERROR: {str(e)}\\n{traceback.format_exc()}"
        with open(input_file + ".error", "w") as f:
            f.write(error_msg)
        print(error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile_path = os.path.join(tmpdir, "Dockerfile")
            runner_path = os.path.join(tmpdir, "runner.py")

            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)

            with open(runner_path, "w") as f:
                f.write(runner_content)

            try:
                self.client.images.build(path=tmpdir, tag=self.image_name, rm=True)
                return True
            except Exception:
                return False

    def run_algorithm(
        self,
        algorithm_file_path: str,
        algorithm_class_name: str,
        algorithm_config: Dict[str, Any],
        simulation_data: Dict[int, Any],
        timeout: int = None,
    ) -> Optional[Dict[int, Any]]:
        if not self.is_available() or not self.ensure_image():
            return None

        if timeout is None:
            timeout = settings.CONTAINER_TIMEOUT_SECONDS

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_input:
            input_data = {
                "algorithm_path": algorithm_file_path,
                "algorithm_class_name": algorithm_class_name,
                "algorithm_config": self._make_serializable(algorithm_config),
                "simulation_data": self._make_serializable(simulation_data),
            }
            pickle.dump(input_data, tmp_input)
            input_filename = tmp_input.name

        try:
            mem_limit = settings.CONTAINER_MEMORY_LIMIT
            cpu_limit = settings.CONTAINER_CPU_LIMIT

            container = self.client.containers.run(
                self.image_name,
                [input_filename],
                mem_limit=mem_limit,
                nano_cpus=int(cpu_limit * 1e9),
                network_disabled=True,
                remove=False,
                detach=True,
            )

            result = container.wait(timeout=timeout)

            if result["StatusCode"] != 0:
                logs = container.logs().decode("utf-8")
                container.remove()
                return None

            output_file = input_filename + ".out"
            error_file = input_filename + ".error"

            if os.path.exists(error_file):
                with open(error_file, "r") as f:
                    error_msg = f.read()
                container.remove()
                os.unlink(input_filename)
                os.unlink(error_file)
                if os.path.exists(output_file):
                    os.unlink(output_file)
                return None

            if not os.path.exists(output_file):
                container.remove()
                os.unlink(input_filename)
                return None

            with open(output_file, "rb") as f:
                result_data = pickle.load(f)

            container.remove()
            os.unlink(input_filename)
            os.unlink(output_file)

            return result_data

        except docker.errors.APIError:
            if os.path.exists(input_filename):
                os.unlink(input_filename)
            return None
        except Exception:
            if os.path.exists(input_filename):
                os.unlink(input_filename)
            return None

    def _make_serializable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif hasattr(obj, "tolist"):
            return obj.tolist()
        else:
            return obj


container_manager = ContainerManager()
