from django.shortcuts import render

import os, joblib
from .modelo.regresion_logistica_logica import funcion_logica_regresion_logistica
from .forms import FormDatosCancer
from regresion.forms import FormSubirArchivo

# Create your views here.
def funcion_regresion_logistica(request):
    resultado = None

    if request.method == 'POST':
        form = FormDatosCancer(request.POST)
        if form.is_valid():
            ruta = os.path.join(os.path.dirname(__file__), 'modelo/modelo_regresion_logistica.pkl')
            modelo = joblib.load(ruta)
            datos = [form.cleaned_data[col] for col in form.fields]
            prediccion = modelo.predict([datos])
            print(prediccion)
            resultado = f'{prediccion[0]}'
            print(resultado)
    else:
        form = FormDatosCancer()

    context = {
        'form': form,
        'resultado': resultado
    }
    return render(request, 'form_cancer.html', context)

def funcion_regresion_logistica_entrenar(request):
    estatus = None
    resultados = None
    form = FormSubirArchivo()

    if request.method == 'POST':
        if request.POST.get('entrenar', None):
            resultados = funcion_logica_regresion_logistica()
        else:
            form = FormSubirArchivo(request.POST, request.FILES)
            if form.is_valid():
                dataset = os.path.join(os.path.dirname(__file__), f"modelo/{request.FILES['archivo']}")
                with open(dataset, 'wb+') as destination:
                    for chunk in request.FILES['archivo'].chunks():
                        destination.write(chunk)
                estatus = 'Se subió con éxito el dataset'

    context = {
        'form':form,
        'estatus': estatus,
        'resultados': resultados,
    }
    return render(request, 'regresion_logistica_entrenamiento.html', context)
