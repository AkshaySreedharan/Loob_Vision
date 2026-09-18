import sys
from django.apps import AppConfig


class ClassifierConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'classifier'

    def ready(self):
        # Pre-load model at app initialization unless running administrative tasks (e.g. migrate, makemigrations)
        ignored_cmds = {'makemigrations', 'migrate', 'collectstatic', 'help'}
        if any(cmd in sys.argv for cmd in ignored_cmds):
            return

        try:
            from classifier.services.predictor import get_predictor
            get_predictor()
        except Exception as e:
            print(f"[ClassifierConfig] Model pre-loading deferred or encountered notice: {e}")
