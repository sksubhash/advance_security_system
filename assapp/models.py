from django.db import models


# Create your models here.
class tbldata(models.Model):
    name = models.CharField(max_length=254)
    mobile_number = models.CharField(max_length=10)
    gender = models.CharField(max_length=10)
    flat_no = models.SmallIntegerField()
    reason = models.TextField()
    type = models.CharField(max_length=10)

    class Meta:
        db_table = "tbldata"


class tblvdetails(models.Model):
    tbldata_id = models.CharField(max_length=254)
    date = models.CharField(max_length=10)
    no_of_time = models.CharField(max_length=256)

    class Meta:
        db_table = "tblvdetails"
