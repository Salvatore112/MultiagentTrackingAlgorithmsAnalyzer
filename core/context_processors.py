from django.utils.translation import get_language


def language_processor(request):
    return {
        'LANGUAGE_CODE': get_language(),
    }