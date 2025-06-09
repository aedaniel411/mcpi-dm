from django import forms

class FormRegresion(forms.Form):
    """
    Formulario para la regresión lineal.
    """
    rm = forms.FloatField(label='RM', required=True, min_value=0.0)
    lstat = forms.FloatField(label='LSTAT', required=True, min_value=0.0)
    piratio = forms.FloatField(label='PTRATIO', required=True, min_value=-100.0, max_value=100.0)

    