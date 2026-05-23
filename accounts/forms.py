import re

from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import User, CustomAlgorithm, SimulationConfig


class RegisterForm(UserCreationForm):
    username = forms.CharField(
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "Username"}
        )
    )
    password1 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={"class": "form-control", "placeholder": "Password"}
        )
    )
    password2 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={"class": "form-control", "placeholder": "Confirm Password"}
        )
    )

    class Meta:
        model = User
        fields = ("username",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs["class"] = "form-control"


class LoginForm(AuthenticationForm):
    username = forms.CharField(
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "Username"}
        )
    )
    password = forms.CharField(
        widget=forms.PasswordInput(
            attrs={"class": "form-control", "placeholder": "Password"}
        )
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs["class"] = "form-control"


class CustomAlgorithmForm(forms.ModelForm):
    class Meta:
        model = CustomAlgorithm
        fields = ("name", "description", "file")
        widgets = {
            "name": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "Algorithm name"}
            ),
            "description": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 3,
                    "placeholder": "Algorithm description (optional)",
                }
            ),
            "file": forms.FileInput(attrs={"class": "form-control", "accept": ".py"}),
        }

    def clean_name(self):
        name = self.cleaned_data.get("name")
        if name:
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
                raise forms.ValidationError(
                    "Algorithm name must be a valid Python identifier (start with letter/underscore, contain only letters, numbers, underscores)."
                )
        return name

    def clean_file(self):
        file = self.cleaned_data.get("file")
        if file:
            try:
                content = file.read().decode("utf-8")
                if "TrackingAlgorithm" not in content:
                    raise forms.ValidationError(
                        "Algorithm must import and inherit from TrackingAlgorithm"
                    )
                file.seek(0)
            except Exception as e:
                raise forms.ValidationError(f"Invalid Python file: {e}")
        return file


class RenameAlgorithmForm(forms.Form):
    new_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "New algorithm name"}
        ),
    )

    def clean_new_name(self):
        name = self.cleaned_data.get("new_name")
        if name:
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
                raise forms.ValidationError(
                    "Algorithm name must be a valid Python identifier (start with letter/underscore, contain only letters, numbers, underscores)."
                )
        return name


class SimulationConfigForm(forms.ModelForm):
    adjacency_sparsity = forms.FloatField(
        required=False,
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": 1}),
        help_text="100% = fully connected, 0% = no connections (except self)",
    )

    class Meta:
        model = SimulationConfig
        fields = (
            "name",
            "description",
            "duration",
            "num_sensors",
            "num_linear_targets",
            "num_random_targets",
            "num_runs",
            "algorithms",
            "noise_enabled",
            "noise_type",
            "noise_low",
            "noise_high",
            "noise_mean",
            "noise_std",
            "adjacency_sparsity",
        )
        widgets = {
            "name": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "Configuration name"}
            ),
            "description": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 2,
                    "placeholder": "Description (optional)",
                }
            ),
            "duration": forms.NumberInput(attrs={"class": "form-control", "min": 1}),
            "num_sensors": forms.NumberInput(attrs={"class": "form-control", "min": 1}),
            "num_linear_targets": forms.NumberInput(
                attrs={"class": "form-control", "min": 0}
            ),
            "num_random_targets": forms.NumberInput(
                attrs={"class": "form-control", "min": 0}
            ),
            "num_runs": forms.NumberInput(attrs={"class": "form-control", "min": 1}),
            "algorithms": forms.SelectMultiple(
                attrs={"class": "form-control", "size": 5}
            ),
            "noise_type": forms.Select(attrs={"class": "form-control"}),
            "noise_low": forms.NumberInput(
                attrs={"class": "form-control", "step": 0.01}
            ),
            "noise_high": forms.NumberInput(
                attrs={"class": "form-control", "step": 0.01}
            ),
            "noise_mean": forms.NumberInput(
                attrs={"class": "form-control", "step": 0.01}
            ),
            "noise_std": forms.NumberInput(
                attrs={"class": "form-control", "step": 0.01}
            ),
        }

    def __init__(self, *args, **kwargs):
        user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)
        if user:
            available_algorithms = [
                "original_spsa",
                "accelerated_spsa",
                "distributed_kalman_filter",
            ]
            custom_algorithms = list(
                user.algorithms.filter(is_active=True).values_list("name", flat=True)
            )
            all_algorithms = available_algorithms + custom_algorithms
            self.fields["algorithms"].choices = [
                (algo, algo) for algo in all_algorithms
            ]

        self.fields["adjacency_sparsity"].widget.attrs[
            "placeholder"
        ] = "100 (fully connected)"
