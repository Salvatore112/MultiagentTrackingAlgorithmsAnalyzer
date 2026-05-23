from django.urls import path
from . import views

app_name = "simulations"

urlpatterns = [
    path("", views.setup_view, name="setup"),
    path("results/", views.results_view, name="results"),
    path(
        "comparison-results/", views.comparison_results_view, name="comparison_results"
    ),
]
