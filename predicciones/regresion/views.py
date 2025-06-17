from django.shortcuts import render
from .forms import FormRegresion
import joblib
import os

# Create your views here.
def funcion_hola(request):
    return render(request, 'hola.html')

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