# Generated by Django 3.0.6 on 2020-10-01 15:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('assapp', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='tbldata',
            name='reason',
            field=models.TextField(default='null'),
            preserve_default=False,
        ),
    ]
