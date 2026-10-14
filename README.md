# Puki3D

Aplicación de escritorio para crear proyectos y modelar objetos 3D. Incluye gestión de proyectos `.puki3d`, un viewport OpenGL con primitivas, selección, transformaciones, outliner e inspector, y guardado de escenas editable.

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
