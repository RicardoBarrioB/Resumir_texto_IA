from django.views.generic import TemplateView
from django.shortcuts import render

class HomeView(TemplateView):
    template_name = 'ia/home.html'

    def post(self, request, *args, **kwargs):
        texto = request.POST.get('texto', '')  # Obtener el texto del formulario
        # Aqui hay que hacer el resumen del texto
        return render(request, self.template_name, {'texto': texto})

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

