# Puki3D

Aplicación de escritorio en Python para crear y editar modelos 3D desde cero, con especial atención a vehículos, accesorios y otros objetos compatibles con American Truck Simulator.

## Requisitos

- Windows 11
- Python 3.11 o posterior

## Preparar el entorno

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Comprobar el entorno

```powershell
puki3d
```

El comando comprueba e informa las versiones de Python, PySide6, PyOpenGL y NumPy. La interfaz y las herramientas 3D todavía no están implementadas.
