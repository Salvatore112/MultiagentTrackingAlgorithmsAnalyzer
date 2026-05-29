import os
import sys
import tempfile
import shutil
import importlib.util
from typing import Dict, Any, Optional
from django.core.files.uploadedfile import UploadedFile
from .container_manager import container_manager


class AlgorithmExecutor:
    def __init__(self, use_container: bool = True):
        self.use_container = use_container and container_manager.is_available()

    def execute_algorithm(
        self,
        algorithm_file,
        algorithm_class_name: str,
        algorithm_config: Dict[str, Any],
        simulation_data: Dict[int, Any],
    ) -> Optional[Dict[int, Any]]:
        if self.use_container:
            return self._execute_in_container(
                algorithm_file, algorithm_class_name, algorithm_config, simulation_data
            )
        else:
            return self._execute_direct(
                algorithm_file, algorithm_class_name, algorithm_config, simulation_data
            )

    def _execute_in_container(
        self,
        algorithm_file,
        algorithm_class_name: str,
        algorithm_config: Dict[str, Any],
        simulation_data: Dict[int, Any],
    ) -> Optional[Dict[int, Any]]:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp_file:
            if hasattr(algorithm_file, "read"):
                algorithm_file.seek(0)
                content = algorithm_file.read()
                if isinstance(content, bytes):
                    tmp_file.write(content)
                else:
                    tmp_file.write(content.encode("utf-8"))
            else:
                with open(algorithm_file, "rb") as f:
                    tmp_file.write(f.read())
            tmp_file_path = tmp_file.name

        try:
            result = container_manager.run_algorithm(
                tmp_file_path,
                algorithm_class_name,
                algorithm_config,
                simulation_data,
            )
            return result
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    def _execute_direct(
        self,
        algorithm_file,
        algorithm_class_name: str,
        algorithm_config: Dict[str, Any],
        simulation_data: Dict[int, Any],
    ) -> Optional[Dict[int, Any]]:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp_file:
            if hasattr(algorithm_file, "read"):
                algorithm_file.seek(0)
                content = algorithm_file.read()
                if isinstance(content, bytes):
                    tmp_file.write(content)
                else:
                    tmp_file.write(content.encode("utf-8"))
            else:
                with open(algorithm_file, "rb") as f:
                    tmp_file.write(f.read())
            tmp_file_path = tmp_file.name

        module_name = (
            f"temp_algorithm_{os.path.basename(tmp_file_path).replace('.', '_')}"
        )

        try:
            spec = importlib.util.spec_from_file_location(module_name, tmp_file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            algorithm_class = None
            for name, obj in module.__dict__.items():
                if name == algorithm_class_name:
                    algorithm_class = obj
                    break

            if algorithm_class is None:
                for name, obj in module.__dict__.items():
                    if isinstance(obj, type) and hasattr(obj, "run_n_iterations"):
                        algorithm_class = obj
                        break

            if algorithm_class is None:
                return None

            algorithm_instance = algorithm_class(**algorithm_config)
            result = algorithm_instance.run_n_iterations(simulation_data)

            return result

        except Exception:
            return None
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            if module_name in sys.modules:
                del sys.modules[module_name]


algorithm_executor = AlgorithmExecutor(use_container=True)
