"""
URL configuration for predicciones project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from regresion import views
import regresion_logistica.views as viewsrl

urlpatterns = [
    path('admin/', admin.site.urls),
    path('hola/', views.funcion_hola),

    # página de bienvenida
    path('', views.home, name='home'),

    # regresión lineal
    path('regresion-lineal/', views.funcion_regresion, name='regresion_lineal'),
    path('regresion-lineal-entrenar/', views.funcion_regresion_entrenar, name='regresion_lineal_entrenar'),

    # regresión lógistica
    path('regresion-logistica/', viewsrl.funcion_regresion_logistica, name='regresion_logistica'),
    path('regresion-logistica-entrenar/', viewsrl.funcion_regresion_logistica_entrenar, name='regresion_logistica_entrenar'),



]
