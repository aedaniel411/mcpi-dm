from django.shortcuts import render
from .forms import FormRegresion, FormSubirArchivo
from .modelo.regresion_boston_housing import regresion
import joblib
import os

# Create your views here.
def funcion_hola(request):
    return render(request, 'hola.html')

def home(request):
    return render(request, 'home.html')

def funcion_regresion_entrenar(request):
    estatus = None
    resultados = None
    form = FormSubirArchivo()

    if request.method == 'POST':
        print(request.POST)
        if request.POST.get('entrenar', None):
            resultados = regresion()
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
    return render(request, 'entrenar_regresion_lineal.html', context)


def funcion_regresion(request):
    form = FormRegresion()
    resultado = None

    if request.method == 'POST':
        # Si el método es POST, significa que se ha enviado el formulario  
        form = FormRegresion(request.POST)
        if form.is_valid():
            ruta = os.path.join(os.path.dirname(__file__), 'modelo/modelo_regresion_lineal.pkl')
            modelo = joblib.load(ruta)
            datos = [form.cleaned_data[col] for col in form.fields]
            prediccion = modelo.predict([datos])
            print(prediccion)
            resultado = f'{prediccion[0]:.2f}'

    context = {
        'form': form,
        'resultado': resultado
    }
    return render(request, 'form_regresion.html', context)