from django.urls import path
from assapp import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login', views.admin_login, name='login'),
    path('live', views.watch_live, name='watch live'),
    path('add_visitor', views.add_visitor, name='add visitor'),
    path('add_residence', views.add_residence, name='add residence'),
    path('add_visitor', views.add_visitor, name='add visitor'),
    path('live', views.watch_live, name='watch live'),
    path('vdetails', views.vdetails, name='view details'),
]
