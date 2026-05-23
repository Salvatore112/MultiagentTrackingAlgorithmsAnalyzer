# Generated migration file
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0002_simulationconfig"),
    ]

    operations = [
        migrations.AddField(
            model_name="simulationconfig",
            name="adjacency_matrix",
            field=models.JSONField(blank=True, default=None, null=True),
        ),
        migrations.AddField(
            model_name="simulationconfig",
            name="adjacency_sparsity",
            field=models.FloatField(blank=True, default=100.0, null=True),
        ),
    ]
