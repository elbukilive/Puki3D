"""Native, versioned project documents for Puki3D."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import unicodedata
from uuid import uuid4


FORMAT_ID = "puki3d.project"
SCHEMA_VERSION = 1
PROJECT_EXTENSION = ".puki3d"
PROJECT_TYPES = {
    "vehicle": "Vehículo",
    "accessory": "Accesorio",
    "object": "Objeto 3D",
}


class ProjectError(ValueError):
    """Raised when a project cannot be created, read, or saved."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_name).strip("-").lower()
    return slug or "proyecto"


@dataclass
class Project:
    """Editable project data, kept independent from the future viewport."""

    path: Path
    data: dict

    @property
    def name(self) -> str:
        return self.data["project"]["name"]

    @property
    def kind(self) -> str:
        return self.data["project"]["type"]

    @property
    def project_id(self) -> str:
        return self.data["project"]["id"]

    @classmethod
    def create(cls, root: Path, name: str, kind: str) -> "Project":
        name = name.strip()
        if not name:
            raise ProjectError("Escribe un nombre para el proyecto.")
        if kind not in PROJECT_TYPES:
            raise ProjectError("El tipo de proyecto no es válido.")
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{_slug(name)}{PROJECT_EXTENSION}"
        if path.exists():
            raise ProjectError("Ya existe un proyecto con ese nombre.")
        stamp = _now()
        data = {
            "format": FORMAT_ID,
            "schema_version": SCHEMA_VERSION,
            "project": {
                "id": str(uuid4()),
                "name": name,
                "type": kind,
                "created_at": stamp,
                "updated_at": stamp,
            },
            "scenes": [
                {"id": str(uuid4()), "name": "Escena principal", "nodes": []}
            ],
            "assets": [],
            "settings": {},
            "editor": {"active_scene": 0},
        }
        project = cls(path, data)
        project.save()
        return project

    @classmethod
    def load(cls, path: Path) -> "Project":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectError(f"No se pudo abrir el proyecto: {exc}") from exc
        if not isinstance(data, dict) or data.get("format") != FORMAT_ID:
            raise ProjectError("Este archivo no es un proyecto válido de Puki3D.")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ProjectError("La versión de este proyecto no es compatible.")
        project_data = data.get("project")
        if not isinstance(project_data, dict) or not project_data.get("name") or project_data.get("type") not in PROJECT_TYPES:
            raise ProjectError("Los datos del proyecto están incompletos o dañados.")
        if not isinstance(data.get("scenes"), list) or not isinstance(data.get("assets"), list):
            raise ProjectError("La estructura de escenas o recursos no es válida.")
        return cls(path, data)

    def save(self) -> None:
        self.data["project"]["updated_at"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(self.data, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, self.path)
        except OSError as exc:
            temp_path.unlink(missing_ok=True)
            raise ProjectError(f"No se pudo guardar el proyecto: {exc}") from exc

    def rename(self, name: str) -> None:
        name = name.strip()
        if not name:
            raise ProjectError("Escribe un nombre para el proyecto.")
        destination = self.path.with_name(f"{_slug(name)}{PROJECT_EXTENSION}")
        if destination != self.path and destination.exists():
            raise ProjectError("Ya existe un proyecto con ese nombre.")
        old_name = self.name
        self.data["project"]["name"] = name
        old_path = self.path
        self.path = destination
        try:
            self.save()
            if old_path != destination:
                old_path.unlink(missing_ok=True)
        except ProjectError:
            self.path = old_path
            self.data["project"]["name"] = old_name
            raise
