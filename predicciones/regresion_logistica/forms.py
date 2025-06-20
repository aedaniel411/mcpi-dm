from django import forms
from django.core.validators import FileExtensionValidator



class FormDatosCancer(forms.Form):
    def __init__(self, *args, **kwargs):
        campos = [
            'mean_radius',
            'mean_texture',
            'mean_perimeter',
            'mean_area',
            'mean_smoothness',
            'mean_compactness',
            'mean_concavity',
            'mean_concave_points',
            'mean_symmetry',
            'mean_fractal_dimension',
            'radius_error',
            'texture_error',
            'perimeter_error',
            'area_error',
            'smoothness_error',
            'compactness_error',
            'concavity_error',
            'concave_points_error',
            'symmetry_error',
            'fractal_dimension_error',
            'worst_radius',
            'worst_texture',
            'worst_perimeter',
            'worst_area',
            'worst_smoothness',
            'worst_compactness',
            'worst_concavity',
            'worst_concave_points',
            'worst_symmetry',
            'worst_fractal_dimension'
        ]
        super().__init__(*args, **kwargs)
        for campo in campos:
            self.fields[campo] = forms.FloatField(widget=forms.NumberInput(attrs={'class':'form-control'}))

