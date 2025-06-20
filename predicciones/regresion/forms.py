from django import forms
from django.core.validators import FileExtensionValidator

class FormSubirArchivo(forms.Form):
    archivo = forms.FileField(
        label='Dataset (csv, xls, xlsx)',
        validators=[FileExtensionValidator(['csv','xls','xlsx'], 'Formato inválido', 'archivo_invalido')],
        widget=forms.FileInput(attrs={'class':'form-control'})
    )

class FormRegresion(forms.Form):
    rm = forms.FloatField(label='RM', widget=forms.NumberInput(attrs={'class':'form-control'}))
    lstat = forms.FloatField(label='LSTAT', widget=forms.NumberInput(attrs={'class':'form-control'}))
    ptratio = forms.FloatField(label='PTRATIO', widget=forms.NumberInput(attrs={'class':'form-control'}))
    

