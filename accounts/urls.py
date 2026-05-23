from django.urls import path
from . import views

app_name = "accounts"

urlpatterns = [
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("register/", views.register_view, name="register"),
    path("profile/", views.profile_view, name="profile"),
    path("algorithms/upload/", views.upload_algorithm, name="upload_algorithm"),
    path(
        "algorithms/<uuid:algorithm_id>/delete/",
        views.delete_algorithm,
        name="delete_algorithm",
    ),
    path(
        "algorithms/<uuid:algorithm_id>/rename/",
        views.rename_algorithm,
        name="rename_algorithm",
    ),
    path("configs/save/", views.save_config, name="save_config"),
    path("configs/<uuid:config_id>/delete/", views.delete_config, name="delete_config"),
    path("configs/<uuid:config_id>/run/", views.run_config, name="run_config"),
    path(
        "configs/run-multiple/", views.run_multiple_configs, name="run_multiple_configs"
    ),
]
