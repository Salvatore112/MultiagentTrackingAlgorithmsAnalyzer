import os
import uuid
import importlib.util
import sys
import random

from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.validators import FileExtensionValidator


class User(AbstractUser):
    email = models.EmailField(unique=True, blank=True, null=True)

    def __str__(self):
        return self.username


class CustomAlgorithm(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="algorithms")
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    file = models.FileField(
        upload_to="algorithms/",
        validators=[FileExtensionValidator(allowed_extensions=["py"])],
    )
    module_name = models.CharField(max_length=200, blank=True, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        unique_together = ["user", "name"]
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} (by {self.user.username})"

    def filename(self):
        return os.path.basename(self.file.name)

    def save(self, *args, **kwargs):
        self.module_name = f"custom_algo_{self.user.id}_{self.id}_{self.name}"
        import re

        self.module_name = re.sub(r"[^a-zA-Z0-9_]", "_", self.module_name)

        super().save(*args, **kwargs)

        self.load_module()

    def load_module(self):
        try:
            file_path = self.file.path
            module_name = self.module_name

            if module_name in sys.modules:
                del sys.modules[module_name]

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            return module
        except Exception as e:
            print(f"Error loading algorithm {self.name}: {e}")
            return None

    def get_algorithm_class(self):
        module = self.load_module()
        if module:
            import inspect
            from algorithms.tracking_algorithm import TrackingAlgorithm

            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, TrackingAlgorithm)
                    and obj != TrackingAlgorithm
                ):
                    return obj
        return None

    def get_algorithm_class_name(self):
        algorithm_class = self.get_algorithm_class()
        if algorithm_class:
            return algorithm_class.__name__
        return None

    def delete(self, *args, **kwargs):
        if self.module_name in sys.modules:
            del sys.modules[self.module_name]

        if self.file:
            self.file.delete(save=False)
        super().delete(*args, **kwargs)


class SimulationConfig(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="configs")
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)

    duration = models.FloatField(default=50)
    num_sensors = models.IntegerField(default=3)
    num_linear_targets = models.IntegerField(default=2)
    num_random_targets = models.IntegerField(default=2)
    num_runs = models.IntegerField(default=1)

    algorithms = models.JSONField(default=list)

    noise_enabled = models.BooleanField(default=False)
    noise_type = models.CharField(max_length=20, default="uniform")
    noise_low = models.FloatField(default=-0.1)
    noise_high = models.FloatField(default=0.1)
    noise_mean = models.FloatField(default=0.0)
    noise_std = models.FloatField(default=0.1)

    lline_config = models.JSONField(default=dict)
    adjacency_matrix = models.JSONField(default=None, null=True, blank=True)
    adjacency_sparsity = models.FloatField(default=100.0, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        unique_together = ["user", "name"]

    def __str__(self):
        return f"{self.name} (by {self.user.username})"

    def generate_adjacency_matrix(self):
        if self.adjacency_matrix is not None:
            return self.adjacency_matrix

        if self.adjacency_sparsity is not None and self.adjacency_sparsity < 100:
            sparsity = self.adjacency_sparsity / 100.0
            matrix = []
            for i in range(self.num_sensors):
                row = []
                for j in range(self.num_sensors):
                    if i == j:
                        row.append(0)
                    else:
                        if random.random() < sparsity:
                            row.append(1)
                        else:
                            row.append(0)
                matrix.append(row)

            for i in range(self.num_sensors):
                for j in range(self.num_sensors):
                    if matrix[i][j] == 1:
                        matrix[j][i] = 1

            return matrix

        return [
            [1 if i != j else 0 for j in range(self.num_sensors)]
            for i in range(self.num_sensors)
        ]

    def to_params_dict(self):
        return {
            "duration": self.duration,
            "num_sensors": self.num_sensors,
            "num_linear_targets": self.num_linear_targets,
            "num_random_targets": self.num_random_targets,
            "algorithms": self.algorithms,
            "noise_enabled": self.noise_enabled,
            "noise_type": self.noise_type,
            "noise_low": self.noise_low,
            "noise_high": self.noise_high,
            "noise_mean": self.noise_mean,
            "noise_std": self.noise_std,
            "num_runs": self.num_runs,
            "lline_config": self.lline_config,
            "adjacency_matrix": self.generate_adjacency_matrix(),
        }
