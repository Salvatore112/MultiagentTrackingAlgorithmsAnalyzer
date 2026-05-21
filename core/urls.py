from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect
from django.http import HttpRequest, HttpResponse
from django.conf import settings
from django.conf.urls.static import static
from django.views.i18n import set_language


def redirect_to_setup(request: HttpRequest) -> HttpResponse:
    return redirect("/setup/")


urlpatterns: list = [
    path("admin/", admin.site.urls),
    path("", redirect_to_setup),
    path("setup/", include("simulations.urls")),
    path("simulation/", include("simulations.urls")),
    path("accounts/", include("accounts.urls")),
    path("i18n/", include("django.conf.urls.i18n")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)