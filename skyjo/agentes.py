"""Interfaz de agente: agente(vista) -> acción (0-25). Un asiento puede ocuparlo un humano, un modelo o un bot."""
import random


class Agente:
    nombre = "agente"

    def decidir(self, vista):
        raise NotImplementedError

    def __call__(self, vista):
        return self.decidir(vista)


class AgenteHumano(Agente):
    """Humano a través de una función pedir(vista) -> acción (consola, web, tests...)."""
    def __init__(self, pedir, nombre="humano"):
        self.pedir, self.nombre = pedir, nombre

    def decidir(self, vista):
        return int(self.pedir(vista))


class ConRuido(Agente):
    """Con probabilidad epsilon juega una acción legal al azar (suaviza un nivel sin entrenar otro modelo)."""
    def __init__(self, base, epsilon):
        self.base, self.epsilon = base, float(epsilon)
        self.nombre = getattr(base, 'nombre', 'agente')

    def decidir(self, vista):
        if random.random() < self.epsilon:
            return int(random.choice(vista.acciones_legales()))
        return int(self.base(vista))
