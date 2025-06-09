from django.shortcuts import render
from .forms import FormRegresion

# Create your views here.
def funcion_hola(request):
    return render(request, 'hola.html')

def funcion_regresion(request):
    form = FormRegresion()
    context = {
        'form': form
    }
    return render(request, 'form_regresion.html', context)