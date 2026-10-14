# Puki3D

Aplicación de escritorio para organizar proyectos de vehículos, accesorios y objetos 3D. La etapa PROJECT incluye creación y apertura de proyectos, guardado en formato propio y un workspace preparado para futuras escenas y herramientas.

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

## Iniciar la aplicación

```powershell
puki3d
```

Los proyectos nuevos se guardan automáticamente en `Documents\Puki3D Projects` como archivos `.puki3d` legibles en JSON y versionados con un esquema propio.
