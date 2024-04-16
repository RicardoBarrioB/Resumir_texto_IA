from django.db import models

class TextoEntrada(models.Model):
    texto = models.TextField()
    fecha_creacion = models.DateTimeField(auto_now_add=True)


class Resumen(models.Model):
    texto_entrada = models.OneToOneField(TextoEntrada, on_delete=models.CASCADE)
    resumen = models.TextField()
    fecha_creacion = models.DateTimeField(auto_now_add=True)




