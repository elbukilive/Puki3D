"""PySide6 desktop interface for the PROJECT and MODEL stages."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QStandardPaths, QSignalBlocker, QTimer
from PySide6.QtGui import QAction, QColor, QShortcut, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QMenu,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .project import PROJECT_EXTENSION, PROJECT_TYPES, Project, ProjectError
from .scene import MeshTopologyError, PRIMITIVES, create_node
from .viewport import SceneViewport


BACKGROUND = "#101114"
PANEL = "#17191e"
PANEL_ALT = "#1d2026"
EDGE = "#2b2e36"
MUTED = "#8d929e"
TEXT = "#eceef2"
ACCENT = "#d5ff58"


def projects_root() -> Path:
    documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
    return Path(documents) / "Puki3D Projects"


class CreateProjectDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Crear proyecto")
        self.setModal(True)
        self.setFixedWidth(470)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 26, 28, 24)
        layout.setSpacing(14)
        title = QLabel("Un nuevo espacio de trabajo")
        title.setObjectName("dialogTitle")
        layout.addWidget(title)
        intro = QLabel("Empieza con un proyecto limpio. Podrás organizar escenas y recursos a medida que avances.")
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        layout.addSpacing(6)
        layout.addWidget(QLabel("NOMBRE DEL PROYECTO"), 0)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Ej. Camión de reparto")
        self.name.setMinimumHeight(42)
        layout.addWidget(self.name)
        layout.addWidget(QLabel("TIPO DE PROYECTO"), 0)
        self.kind = QComboBox()
        for key, label in PROJECT_TYPES.items():
            self.kind.addItem(label, key)
        self.kind.setMinimumHeight(42)
        layout.addWidget(self.kind)
        hint = QLabel(f"Se guardará automáticamente en {projects_root()}")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Crear proyecto")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancelar")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addSpacing(8)
        layout.addWidget(buttons)
        self.name.setFocus()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Puki3D")
        self.resize(1440, 900)
        self.setMinimumSize(1050, 680)
        self.project: Project | None = None
        self.settings = QApplication.instance().organizationName()
        self._setup_menu()
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.home = self._build_home()
        self.stack.addWidget(self.home)
        self.workspace = self._build_workspace()
        self.stack.addWidget(self.workspace)
        self._apply_style()
        self._refresh_recent()
        self.show_home()

    def _setup_menu(self) -> None:
        self.setMenuBar(self.menuBar())
        project_menu = self.menuBar().addMenu("Proyecto")
        new_action = QAction("Nuevo proyecto…", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.create_project)
        project_menu.addAction(new_action)
        open_action = QAction("Abrir proyecto…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_project_dialog)
        project_menu.addAction(open_action)
        self.save_action = QAction("Guardar", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.triggered.connect(self.save_project)
        project_menu.addAction(self.save_action)
        self.rename_action = QAction("Renombrar…", self)
        self.rename_action.triggered.connect(self.rename_project)
        project_menu.addAction(self.rename_action)
        project_menu.addSeparator()
        close_action = QAction("Cerrar proyecto", self)
        close_action.setShortcut("Ctrl+W")
        close_action.triggered.connect(self.show_home)
        project_menu.addAction(close_action)
        self._set_project_actions(False)

    def _build_home(self) -> QWidget:
        root = QWidget()
        root.setObjectName("homeRoot")
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(232)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(22, 26, 18, 20)
        side_layout.setSpacing(22)
        side_layout.addWidget(self._brand())
        side_layout.addSpacing(28)
        side_layout.addWidget(self._section_label("ESPACIO DE TRABAJO"))
        side_layout.addWidget(self._nav_item("⌂", "Inicio", selected=True))
        side_layout.addWidget(self._nav_item("▧", "Proyectos"))
        side_layout.addStretch()
        side_layout.addWidget(self._section_label("PUKI3D  ·  ETAPA 2"))
        outer.addWidget(side)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(54, 38, 54, 42)
        content_layout.setSpacing(26)
        header = QHBoxLayout()
        heading = QVBoxLayout()
        eyebrow = QLabel("TU ESPACIO CREATIVO")
        eyebrow.setObjectName("eyebrow")
        heading.addWidget(eyebrow)
        welcome = QLabel("Proyectos")
        welcome.setObjectName("pageTitle")
        heading.addWidget(welcome)
        header.addLayout(heading)
        header.addStretch()
        self.open_button = QPushButton("Abrir proyecto")
        self.open_button.setObjectName("secondaryButton")
        self.open_button.clicked.connect(self.open_project_dialog)
        header.addWidget(self.open_button)
        content_layout.addLayout(header)

        hero = QFrame()
        hero.setObjectName("hero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(34, 30, 34, 28)
        copy = QVBoxLayout()
        kicker = QLabel("BIENVENIDO A PUKI3D")
        kicker.setObjectName("eyebrow")
        copy.addWidget(kicker)
        hero_title = QLabel("Dale forma a tu\npróxima idea.")
        hero_title.setObjectName("heroTitle")
        copy.addWidget(hero_title)
        hero_text = QLabel("Crea un proyecto y prepara el escenario para construir en 3D.")
        hero_text.setObjectName("muted")
        hero_text.setWordWrap(True)
        copy.addWidget(hero_text)
        create = QPushButton("＋   Crear proyecto nuevo")
        create.setObjectName("primaryButton")
        create.setMinimumHeight(44)
        create.clicked.connect(self.create_project)
        copy.addSpacing(10)
        copy.addWidget(create, 0, Qt.AlignmentFlag.AlignLeft)
        copy.addStretch()
        hero_layout.addLayout(copy, 3)
        visual = QFrame()
        visual.setObjectName("heroVisual")
        visual.setMinimumWidth(290)
        visual_layout = QVBoxLayout(visual)
        visual_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cube = QLabel("◇")
        cube.setObjectName("heroGlyph")
        cube.setAlignment(Qt.AlignmentFlag.AlignCenter)
        visual_layout.addWidget(cube)
        visual_caption = QLabel("UN LIENZO EN BLANCO")
        visual_caption.setObjectName("eyebrow")
        visual_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        visual_layout.addWidget(visual_caption)
        hero_layout.addWidget(visual, 2)
        content_layout.addWidget(hero, 2)

        row = QHBoxLayout()
        recent_panel = QFrame()
        recent_panel.setObjectName("card")
        recent_layout = QVBoxLayout(recent_panel)
        recent_layout.setContentsMargins(22, 19, 22, 18)
        title_row = QHBoxLayout()
        recent_title = QLabel("Recientes")
        recent_title.setObjectName("sectionTitle")
        title_row.addWidget(recent_title)
        title_row.addStretch()
        title_row.addWidget(self._text_button("Ver todos", self.open_project_dialog))
        recent_layout.addLayout(title_row)
        self.recent_list = QListWidget()
        self.recent_list.setObjectName("recentList")
        self.recent_list.itemDoubleClicked.connect(self._open_recent_item)
        recent_layout.addWidget(self.recent_list)
        row.addWidget(recent_panel, 3)

        aside = QFrame()
        aside.setObjectName("card")
        aside.setFixedWidth(270)
        aside_layout = QVBoxLayout(aside)
        aside_layout.setContentsMargins(22, 21, 22, 20)
        aside_layout.setSpacing(13)
        next_label = QLabel("PREPARADO PARA CRECER")
        next_label.setObjectName("eyebrow")
        aside_layout.addWidget(next_label)
        aside_title = QLabel("Una base para\nlo que viene")
        aside_title.setObjectName("sectionTitle")
        aside_layout.addWidget(aside_title)
        aside_layout.addWidget(self._feature("01", "Proyectos propios", "Tus datos se guardan en un formato editable."))
        aside_layout.addWidget(self._feature("02", "Escenas organizadas", "Cada proyecto comienza con su escena principal."))
        aside_layout.addWidget(self._feature("03", "Espacio preparado", "Un workspace listo para las siguientes etapas."))
        aside_layout.addStretch()
        row.addWidget(aside)
        content_layout.addLayout(row, 2)
        outer.addWidget(content, 1)
        return root

    def _build_workspace(self) -> QWidget:
        root = QWidget()
        root.setObjectName("workspaceRoot")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        top = QFrame()
        top.setObjectName("topbar")
        top.setFixedHeight(66)
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(24, 0, 24, 0)
        top_layout.addWidget(self._brand(compact=True))
        top_layout.addWidget(self._divider(vertical=True))
        self.project_title = QLabel("Proyecto")
        self.project_title.setObjectName("topProjectTitle")
        top_layout.addWidget(self.project_title)
        self.project_type = QLabel("")
        self.project_type.setObjectName("typePill")
        top_layout.addWidget(self.project_type)
        top_layout.addStretch()
        self.save_state = QLabel("●  Guardado")
        self.save_state.setObjectName("muted")
        top_layout.addWidget(self.save_state)
        add_button = QToolButton()
        add_button.setText("＋ Añadir")
        add_button.setObjectName("secondaryButton")
        add_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.add_menu = QMenu(add_button)
        add_button.setMenu(self.add_menu)
        top_layout.addWidget(add_button)
        rename_button = QPushButton("Renombrar")
        rename_button.setObjectName("quietButton")
        rename_button.clicked.connect(self.rename_project)
        top_layout.addWidget(rename_button)
        save_button = QPushButton("Guardar")
        save_button.setObjectName("primaryButtonSmall")
        save_button.clicked.connect(self.save_project)
        top_layout.addWidget(save_button)
        layout.addWidget(top)

        self.add_menu_bar = self.menuBar().addMenu("Añadir")
        self._add_primitive_actions(self.add_menu_bar)
        self._add_primitive_actions(self.add_menu)
        self.add_menu_bar.setEnabled(False)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        rail = QFrame()
        rail.setObjectName("toolRail")
        rail.setFixedWidth(62)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(8, 16, 8, 12)
        rail_layout.setSpacing(8)
        self.tool_buttons = {}
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        for tool, icon, label in (("select", "↖", "Seleccionar"), ("move", "✥", "Mover"), ("rotate", "⟳", "Rotar"), ("scale", "⤢", "Escalar")):
            button = QPushButton(icon)
            button.setObjectName("toolActive" if tool == "select" else "toolButton")
            button.setToolTip(label)
            button.setAccessibleName(label)
            button.setFixedSize(44, 44)
            button.setCheckable(True)
            button.setChecked(tool == "select")
            self.tool_group.addButton(button)
            button.clicked.connect(lambda checked=False, current=tool: self._set_tool(current))
            self.tool_buttons[tool] = button
            rail_layout.addWidget(button)
        rail_layout.addStretch()
        rail_layout.addWidget(self._divider())
        settings_tool = QPushButton("⚙")
        settings_tool.setObjectName("toolButton")
        settings_tool.setToolTip("Preferencias")
        settings_tool.setFixedSize(44, 44)
        rail_layout.addWidget(settings_tool)
        body.addWidget(rail)

        canvas_area = QWidget()
        canvas_area.setObjectName("canvasArea")
        canvas_layout = QVBoxLayout(canvas_area)
        canvas_layout.setContentsMargins(14, 14, 14, 12)
        canvas_layout.setSpacing(10)
        canvas_toolbar = QHBoxLayout()
        view_label = QLabel("VISTA DE ESCENA")
        view_label.setObjectName("eyebrow")
        canvas_toolbar.addWidget(view_label)
        self.mode_buttons = {}
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for mode, label in (("object", "Objeto"), ("edit", "Edición")):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setObjectName("modeActive" if mode == "object" else "toolButton")
            button.setChecked(mode == "object")
            self.mode_group.addButton(button)
            button.clicked.connect(lambda checked=False, value=mode: self.viewport.set_mode(value))
            self.mode_buttons[mode] = button
            canvas_toolbar.addWidget(button)
        self.component_buttons = {}
        self.component_group = QButtonGroup(self)
        self.component_group.setExclusive(True)
        for mode, label in (("vertex", "V"), ("edge", "A"), ("face", "C")):
            button = QPushButton(label)
            button.setToolTip({"vertex": "Vértices (1)", "edge": "Aristas (2)", "face": "Caras (3)"}[mode])
            button.setCheckable(True)
            button.setChecked(mode == "face")
            button.setEnabled(False)
            self.component_group.addButton(button)
            button.clicked.connect(lambda checked=False, value=mode: self.viewport.set_component_mode(value))
            self.component_buttons[mode] = button
            canvas_toolbar.addWidget(button)
        undo_button = QPushButton("↶")
        undo_button.setToolTip("Deshacer edición (Ctrl+Z)")
        undo_button.clicked.connect(lambda: self.viewport.undo_edit())
        canvas_toolbar.addWidget(undo_button)
        redo_button = QPushButton("↷")
        redo_button.setToolTip("Rehacer edición (Ctrl+Y)")
        redo_button.clicked.connect(lambda: self.viewport.redo_edit())
        canvas_toolbar.addWidget(redo_button)
        self.inset_button = QPushButton("Inset (I)")
        self.inset_button.setToolTip("Crear un borde interior en la cara seleccionada")
        self.inset_button.setEnabled(False)
        self.inset_button.clicked.connect(lambda: self.viewport.inset_selected_face())
        canvas_toolbar.addWidget(self.inset_button)
        mirror_button = QToolButton()
        mirror_button.setText("Mirror")
        mirror_button.setToolTip("Activar o desactivar simetría sobre X, Y o Z")
        mirror_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        mirror_menu = QMenu(mirror_button)
        self.mirror_axis_actions = {}
        for axis_index, axis in enumerate("XYZ"):
            action = QAction(f"Mirror {axis}", self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, value=axis_index: self._toggle_mirror_axis(value))
            mirror_menu.addAction(action)
            self.mirror_axis_actions[axis_index] = action
        mirror_button.setMenu(mirror_menu)
        canvas_toolbar.addWidget(mirror_button)
        canvas_toolbar.addStretch()
        view_mode = QLabel("Perspectiva  ·  3D")
        view_mode.setObjectName("muted")
        canvas_toolbar.addWidget(view_mode)
        canvas_layout.addLayout(canvas_toolbar)
        self.viewport = SceneViewport()
        self.viewport.selectionChanged.connect(self._on_viewport_selection)
        self.viewport.transformChanged.connect(self._on_viewport_transform)
        self.viewport.sceneChanged.connect(self._mark_project_dirty)
        self.viewport.modeChanged.connect(self._on_viewport_mode_changed)
        self.viewport.extrusionStateChanged.connect(self._on_extrusion_state_changed)
        self.viewport.insetStateChanged.connect(self._on_inset_state_changed)
        canvas_layout.addWidget(self.viewport, 1)
        status = QHBoxLayout()
        status_label = QLabel("ESCENA PRINCIPAL")
        self.edit_status = status_label
        status_label.setObjectName("eyebrow")
        status.addWidget(status_label)
        status.addStretch()
        status.addWidget(QLabel("Unidades: metros"))
        canvas_layout.addLayout(status)
        body.addWidget(canvas_area, 1)

        inspector = QFrame()
        inspector.setObjectName("inspector")
        inspector.setFixedWidth(282)
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.setSpacing(0)
        inspector_title = QLabel("ESQUEMA")
        inspector_title.setObjectName("panelHeading")
        inspector_layout.addWidget(inspector_title)
        scene_row = QFrame()
        scene_row.setObjectName("sceneRow")
        scene_row_layout = QHBoxLayout(scene_row)
        scene_row_layout.setContentsMargins(16, 13, 12, 13)
        scene_icon = QLabel("▱")
        scene_icon.setObjectName("sceneIcon")
        scene_row_layout.addWidget(scene_icon)
        scene_name = QLabel("Escena principal")
        scene_row_layout.addWidget(scene_name)
        scene_row_layout.addStretch()
        scene_row_layout.addWidget(QLabel("⌄"))
        inspector_layout.addWidget(scene_row)
        self.outliner = QListWidget()
        self.outliner.setObjectName("outliner")
        self.outliner.currentRowChanged.connect(self._on_outliner_selection)
        inspector_layout.addWidget(self.outliner, 2)
        object_actions = QHBoxLayout()
        object_actions.setContentsMargins(10, 8, 10, 8)
        object_actions.setSpacing(6)
        add_object = QToolButton()
        add_object.setText("＋")
        add_object.setToolTip("Añadir objeto")
        add_object.setObjectName("secondaryButton")
        add_object.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        object_menu = QMenu(add_object)
        self._add_primitive_actions(object_menu)
        add_object.setMenu(object_menu)
        object_actions.addWidget(add_object)
        duplicate_button = QPushButton("⧉")
        duplicate_button.setToolTip("Duplicar objeto (Ctrl+D)")
        duplicate_button.setObjectName("secondaryButton")
        duplicate_button.clicked.connect(self.duplicate_selected)
        object_actions.addWidget(duplicate_button)
        delete_button = QPushButton("⌫")
        delete_button.setToolTip("Eliminar objeto (Supr)")
        delete_button.setObjectName("secondaryButton")
        delete_button.clicked.connect(self.delete_selected)
        object_actions.addWidget(delete_button)
        object_actions.addStretch()
        inspector_layout.addLayout(object_actions)
        inspector_layout.addWidget(self._divider())
        properties_title = QLabel("INSPECTOR")
        properties_title.setObjectName("panelHeading")
        inspector_layout.addWidget(properties_title)
        self.object_hint = QLabel("Selecciona un objeto")
        self.object_hint.setObjectName("muted")
        self.object_hint.setContentsMargins(16, 0, 8, 5)
        inspector_layout.addWidget(self.object_hint)
        self.transform_panel = QWidget()
        self.transform_panel.setObjectName("transformPanel")
        transform_layout = QGridLayout(self.transform_panel)
        transform_layout.setContentsMargins(14, 2, 14, 14)
        transform_layout.setHorizontalSpacing(4)
        transform_layout.setVerticalSpacing(5)
        self.transform_fields = {}
        for axis_index, axis in enumerate("XYZ", start=1):
            axis_label = QLabel(axis)
            axis_label.setObjectName("axisLabel")
            axis_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            transform_layout.addWidget(axis_label, 0, axis_index)
        for row, (key, label) in enumerate((("position", "Posición"), ("rotation", "Rotación"), ("scale", "Escala")), start=1):
            title = QLabel(label)
            title.setObjectName("transformLabel")
            transform_layout.addWidget(title, row, 0)
            fields = []
            for axis_index, axis in enumerate("XYZ", start=1):
                field = QDoubleSpinBox()
                field.setObjectName("transformField")
                field.setDecimals(3)
                field.setSingleStep(0.1 if key != "rotation" else 5.0)
                field.setKeyboardTracking(False)
                field.setRange(0.01, 10000.0) if key == "scale" else field.setRange(-100000.0, 100000.0)
                field.setToolTip(f"{label} {axis}")
                field.valueChanged.connect(self._apply_inspector_transform)
                transform_layout.addWidget(field, row, axis_index)
                fields.append(field)
            self.transform_fields[key] = fields
        inspector_layout.addWidget(self.transform_panel)
        self.transform_panel.setEnabled(False)
        self.mirror_hint = QLabel("ESPEJO: DESACTIVADO")
        self.mirror_hint.setObjectName("muted")
        self.mirror_hint.setContentsMargins(16, 0, 8, 10)
        inspector_layout.addWidget(self.mirror_hint)
        self._on_viewport_mode_changed(self.viewport.mode, self.viewport.component_mode)
        body.addWidget(inspector)
        layout.addLayout(body, 1)

        self.autosave_timer = QTimer(self)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.setInterval(400)
        self.autosave_timer.timeout.connect(self._save_project_silently)
        for tool, shortcut in (("move", "W"), ("rotate", "E"), ("scale", "R"), ("select", "Q")):
            action = QShortcut(shortcut, self.viewport)
            action.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            action.activated.connect(lambda current=tool: self._activate_tool_shortcut(current))
        delete_shortcut = QShortcut("Delete", self.viewport)
        delete_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        delete_shortcut.activated.connect(self.delete_selected)
        duplicate_shortcut = QShortcut("Ctrl+D", self.viewport)
        duplicate_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        duplicate_shortcut.activated.connect(self.duplicate_selected)
        undo_shortcut = QShortcut("Ctrl+Z", self.viewport)
        undo_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        undo_shortcut.activated.connect(self.viewport.undo_edit)
        redo_shortcut = QShortcut("Ctrl+Y", self.viewport)
        redo_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        redo_shortcut.activated.connect(self.viewport.redo_edit)
        redo_shift_shortcut = QShortcut("Ctrl+Shift+Z", self.viewport)
        redo_shift_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        redo_shift_shortcut.activated.connect(self.viewport.redo_edit)
        return root

    def _activate_tool_shortcut(self, tool: str) -> None:
        if tool == "rotate" and self.viewport.mode == "edit":
            self.viewport.extrude_selected_face()
        else:
            self._set_tool(tool)

    def _on_viewport_mode_changed(self, mode: str, component: str) -> None:
        for value, button in self.mode_buttons.items():
            button.setChecked(value == mode)
            button.setObjectName("modeActive" if value == mode else "toolButton")
            button.style().unpolish(button)
            button.style().polish(button)
        for value, button in self.component_buttons.items():
            button.setChecked(value == component)
            button.setEnabled(mode == "edit")
        self.inset_button.setEnabled(mode == "edit" and component == "face" and bool(self.viewport.selected_node))
        self._sync_mirror_controls(self.viewport.selected_node)
        if self.viewport.insetting:
            self.edit_status.setText("INSET ? Arrastra a la derecha para ajustar ? Enter confirma ? Esc cancela")
        elif self.viewport.extruding:
            self.edit_status.setText("Extrusi?n ? Mueve el cursor y confirma con Enter ? Esc cancela")
        elif mode == "edit":
            mirror = self.viewport.selected_node.get("mirror") if self.viewport.selected_node else None
            suffix = f" ? MIRROR {'XYZ'[int(mirror['axis'])]}" if mirror else ""
            self.edit_status.setText("Modo edici?n ? " + {"vertex": "V?rtices", "edge": "Aristas", "face": "Caras"}[component] + suffix)
        else:
            self.edit_status.setText("ESCENA PRINCIPAL")

    def _on_extrusion_state_changed(self, active: bool) -> None:
        self._on_viewport_mode_changed(self.viewport.mode, self.viewport.component_mode)

    def _on_inset_state_changed(self, active: bool) -> None:
        self._on_viewport_mode_changed(self.viewport.mode, self.viewport.component_mode)

    def _toggle_mirror_axis(self, axis: int) -> None:
        try:
            self.viewport.set_mirror_axis(axis)
        except MeshTopologyError as exc:
            self.edit_status.setText(str(exc))
        self._sync_mirror_controls(self.viewport.selected_node)
        self._on_viewport_mode_changed(self.viewport.mode, self.viewport.component_mode)

    def _sync_mirror_controls(self, node) -> None:
        mirror = node.get("mirror") if node else None
        for axis, action in self.mirror_axis_actions.items():
            action.setEnabled(bool(node) and self.viewport.mode == "edit")
            action.setChecked(bool(mirror and int(mirror.get("axis", -1)) == axis))
        if node and mirror:
            self.mirror_hint.setText(f"ESPEJO ACTIVO ? EJE {'XYZ'[int(mirror['axis'])]}")
        else:
            self.mirror_hint.setText("ESPEJO: DESACTIVADO")

    def _add_primitive_actions(self, menu: QMenu) -> None:
        for primitive, label in PRIMITIVES.items():
            action = QAction(label, self)
            action.triggered.connect(lambda checked=False, kind=primitive: self.add_primitive(kind))
            menu.addAction(action)

    def _set_tool(self, tool: str) -> None:
        self.viewport.set_tool(tool)
        for current, button in self.tool_buttons.items():
            button.setObjectName("toolActive" if current == tool else "toolButton")
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()

    def _active_scene_nodes(self) -> list[dict]:
        if not self.project:
            return []
        scenes = self.project.data.get("scenes", [])
        index = self.project.data.get("editor", {}).get("active_scene", 0)
        if not scenes:
            scenes.append({"id": "main", "name": "Escena principal", "nodes": []})
        index = max(0, min(index, len(scenes) - 1))
        return scenes[index].setdefault("nodes", [])

    def _refresh_outliner(self, selected_id: str | None = None) -> None:
        nodes = self._active_scene_nodes()
        self.outliner.blockSignals(True)
        self.outliner.clear()
        selected_row = -1
        for row, node in enumerate(nodes):
            item = QListWidgetItem(f"◈  {node.get('name', 'Objeto')}")
            item.setData(Qt.ItemDataRole.UserRole, node.get("id"))
            self.outliner.addItem(item)
            if node.get("id") == selected_id:
                selected_row = row
        self.outliner.setCurrentRow(selected_row)
        self.outliner.blockSignals(False)
        self.viewport.set_nodes(nodes)
        self.viewport.select_node(selected_id)

    def _on_outliner_selection(self, row: int) -> None:
        item = self.outliner.item(row) if row >= 0 else None
        node_id = item.data(Qt.ItemDataRole.UserRole) if item else None
        self.viewport.select_node(node_id)

    def _on_viewport_selection(self, node) -> None:
        node_id = node.get("id") if node else None
        row = next((index for index in range(self.outliner.count()) if self.outliner.item(index).data(Qt.ItemDataRole.UserRole) == node_id), -1)
        self.outliner.blockSignals(True)
        self.outliner.setCurrentRow(row)
        self.outliner.blockSignals(False)
        self._update_transform_fields(node)
        self._on_viewport_mode_changed(self.viewport.mode, self.viewport.component_mode)

    def _on_viewport_transform(self, node) -> None:
        self._update_transform_fields(node)

    def _update_transform_fields(self, node) -> None:
        self.transform_panel.setEnabled(bool(node))
        self.object_hint.setText(node.get("name", "Objeto") if node else "Selecciona un objeto")
        for key, fields in self.transform_fields.items():
            values = node.get("transform", {}).get(key, [1, 1, 1] if key == "scale" else [0, 0, 0]) if node else ([1, 1, 1] if key == "scale" else [0, 0, 0])
            for field, value in zip(fields, values):
                blocker = QSignalBlocker(field)
                field.setValue(float(value))
                del blocker

    def _apply_inspector_transform(self) -> None:
        node = self.viewport.selected_node
        if not node:
            return
        for key, fields in self.transform_fields.items():
            node.setdefault("transform", {})[key] = [field.value() for field in fields]
        self.viewport.update()
        self._mark_project_dirty()

    def add_primitive(self, primitive: str) -> None:
        if not self.project or primitive not in PRIMITIVES:
            return
        nodes = self._active_scene_nodes()
        ordinal = sum(node.get("primitive") == primitive for node in nodes) + 1
        node = create_node(primitive, ordinal)
        if nodes:
            node["transform"]["position"][0] = len(nodes) * 0.7
        nodes.append(node)
        self._refresh_outliner(node["id"])
        self._mark_project_dirty()

    def duplicate_selected(self) -> None:
        selected = self.viewport.selected_node
        if not self.project or not selected:
            return
        import copy
        from uuid import uuid4
        duplicate = copy.deepcopy(selected)
        duplicate["id"] = str(uuid4())
        duplicate["name"] = f"{selected.get('name', 'Objeto')} copia"
        duplicate["transform"]["position"][0] += 0.75
        nodes = self._active_scene_nodes()
        nodes.append(duplicate)
        self._refresh_outliner(duplicate["id"])
        self._mark_project_dirty()

    def delete_selected(self) -> None:
        if self.viewport.mode == "edit":
            self.viewport.delete_selected_components()
            return
        selected = self.viewport.selected_node
        if not self.project or not selected:
            return
        nodes = self._active_scene_nodes()
        nodes[:] = [node for node in nodes if node.get("id") != selected.get("id")]
        self._refresh_outliner(None)
        self._mark_project_dirty()

    def _mark_project_dirty(self) -> None:
        if self.project:
            self.save_state.setText("●  Guardando…")
            self.autosave_timer.start()

    def _save_project_silently(self) -> None:
        if not self.project:
            return
        try:
            self.project.save()
        except ProjectError as exc:
            self.save_state.setText("●  Error al guardar")
            QMessageBox.critical(self, "No se pudo guardar", str(exc))
            return
        self.save_state.setText("●  Guardado")

    def _brand(self, compact: bool = False) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        mark = QLabel("P")
        mark.setObjectName("brandMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(34, 34)
        layout.addWidget(mark)
        name = QLabel("Puki3D")
        name.setObjectName("brandName")
        layout.addWidget(name)
        if not compact:
            tag = QLabel("MODEL")
            tag.setObjectName("brandTag")
            layout.addWidget(tag)
        return widget

    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        return label

    @staticmethod
    def _nav_item(icon: str, label: str, selected: bool = False) -> QLabel:
        item = QLabel(f"  {icon}     {label}")
        item.setObjectName("navSelected" if selected else "navItem")
        item.setMinimumHeight(42)
        return item

    @staticmethod
    def _text_button(text: str, callback) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("linkButton")
        button.clicked.connect(callback)
        return button

    @staticmethod
    def _feature(number: str, title: str, description: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 6, 0, 5)
        layout.setSpacing(3)
        head = QHBoxLayout()
        index = QLabel(number)
        index.setObjectName("featureIndex")
        head.addWidget(index)
        title_label = QLabel(title)
        title_label.setObjectName("featureTitle")
        head.addWidget(title_label)
        head.addStretch()
        layout.addLayout(head)
        detail = QLabel(description)
        detail.setObjectName("muted")
        detail.setWordWrap(True)
        layout.addWidget(detail)
        return widget

    @staticmethod
    def _divider(vertical: bool = False) -> QFrame:
        line = QFrame()
        line.setObjectName("divider")
        line.setFrameShape(QFrame.Shape.VLine if vertical else QFrame.Shape.HLine)
        return line

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            QMainWindow, QDialog {{ background: {BACKGROUND}; }}
            QWidget {{ background: transparent; color: {TEXT}; font-family: 'Segoe UI'; font-size: 10pt; }}
            QLabel {{ background: transparent; }}
            #homeRoot, #workspaceRoot {{ background: {BACKGROUND}; }}
            QMenuBar {{ background: {BACKGROUND}; color: #aeb3bd; padding-left: 8px; }}
            QMenuBar::item {{ padding: 6px 10px; }}
            QMenuBar::item:selected, QMenu {{ background: {PANEL}; }}
            QMenu::item {{ padding: 7px 28px 7px 18px; }}
            QMenu::item:selected {{ background: {PANEL_ALT}; }}
            #sidebar, #topbar, #toolRail {{ background: {PANEL}; }}
            #sidebar {{ border-right: 1px solid {EDGE}; }}
            #brandMark {{ background: {ACCENT}; color: #171a12; border-radius: 9px; font-weight: 800; font-size: 15pt; }}
            #brandName {{ font-size: 15pt; font-weight: 700; }}
            #brandTag, #sectionLabel, #eyebrow, #panelHeading {{ color: {MUTED}; font-size: 8pt; font-weight: 700; letter-spacing: 1px; }}
            #navItem, #navSelected {{ border-radius: 8px; font-weight: 500; }}
            #navItem {{ color: #9da2ac; }}
            #navSelected {{ background: #272b32; color: {TEXT}; border-left: 2px solid {ACCENT}; }}
            #pageTitle {{ font-size: 25pt; font-weight: 650; }}
            #hero {{ background: #1a1e23; border: 1px solid #30353c; border-radius: 16px; }}
            #heroTitle {{ font-size: 30pt; font-weight: 650; line-height: 1.1; }}
            #heroVisual {{ background: #22272d; border: 1px solid #343a42; border-radius: 12px; }}
            #heroGlyph {{ color: {ACCENT}; font-size: 88pt; }}
            #card {{ background: {PANEL}; border: 1px solid {EDGE}; border-radius: 12px; }}
            #sectionTitle {{ font-size: 13pt; font-weight: 650; }}
            #featureTitle {{ font-weight: 600; }}
            #featureIndex {{ color: {ACCENT}; font-size: 8pt; font-weight: 700; }}
            #muted {{ color: {MUTED}; }}
            #recentList {{ border: 0; background: transparent; outline: 0; }}
            #recentList::item {{ min-height: 47px; border-bottom: 1px solid {EDGE}; padding: 5px 4px; }}
            #recentList::item:selected {{ background: #252a30; border-radius: 7px; }}
            QPushButton {{ border: 0; border-radius: 7px; padding: 8px 13px; font-weight: 600; }}
            #primaryButton, #primaryButtonSmall {{ background: {ACCENT}; color: #1b2012; }}
            #primaryButton:hover, #primaryButtonSmall:hover {{ background: #e2ff8d; }}
            #primaryButton {{ padding: 10px 17px; }}
            #primaryButtonSmall {{ padding: 8px 14px; }}
            #secondaryButton {{ background: #252930; color: {TEXT}; border: 1px solid #343841; }}
            #secondaryButton:hover, #quietButton:hover, #toolButton:hover {{ background: #2b3037; }}
            #quietButton, #linkButton {{ background: transparent; color: #b5bac4; }}
            #linkButton {{ color: {ACCENT}; padding-right: 0; }}
            #dialogTitle {{ font-size: 18pt; font-weight: 650; }}
            QLineEdit, QComboBox {{ background: #111318; border: 1px solid #353943; border-radius: 7px; padding: 8px 10px; selection-background-color: #71843d; }}
            QLineEdit:focus, QComboBox:focus {{ border: 1px solid #a7c94c; }}
            QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {EDGE}; selection-background-color: #30363e; }}
            QDialogButtonBox QPushButton {{ background: #282c33; min-width: 90px; }}
            QDialogButtonBox QPushButton[text="Crear proyecto"] {{ background: {ACCENT}; color: #1b2012; }}
            #topbar {{ border-bottom: 1px solid {EDGE}; }}
            #topProjectTitle {{ font-size: 11pt; font-weight: 650; }}
            #typePill {{ color: #bdc2cc; background: #292d34; border-radius: 10px; padding: 4px 9px; font-size: 8pt; }}
            #divider {{ color: {EDGE}; background: {EDGE}; max-height: 1px; }}
            #inspector {{ background: {PANEL}; border-left: 1px solid {EDGE}; }}
            #panelHeading {{ padding: 17px 16px 12px; }}
            #sceneRow {{ background: #22262d; border-top: 1px solid {EDGE}; border-bottom: 1px solid {EDGE}; }}
            #sceneIcon {{ color: {ACCENT}; }}
            #infoValue {{ padding: 4px 16px 1px; font-weight: 600; }}
            #infoKind {{ padding: 0 16px 5px; }}
            #infoPath {{ padding: 0 16px; font-size: 8pt; }}
            #toolActive {{ background: #343a31; color: {ACCENT}; font-size: 16pt; }}
            #toolButton {{ background: transparent; color: #9ca2ad; font-size: 15pt; }}
            #viewport {{ border: 1px solid #2c3038; border-radius: 7px; }}
            #outliner {{ background: transparent; border: 0; outline: 0; padding: 5px 9px; }}
            #outliner::item {{ min-height: 31px; border-radius: 5px; padding: 4px 8px; }}
            #outliner::item:selected {{ background: #303640; color: {TEXT}; border-left: 2px solid {ACCENT}; }}
            #transformPanel {{ background: transparent; }}
            #transformLabel {{ color: #b8bdc6; font-size: 9pt; }}
            #axisLabel {{ color: {MUTED}; font-size: 8pt; font-weight: 700; }}
            #transformField {{ min-width: 51px; max-width: 65px; padding: 4px 2px; background: #111318; border: 1px solid #353943; border-radius: 5px; }}
            #toolActive:checked {{ background: #343a31; color: {ACCENT}; font-size: 16pt; }}
            #toolButton:checked {{ background: #343a31; color: {ACCENT}; }}
        """)

    def _set_project_actions(self, enabled: bool) -> None:
        self.save_action.setEnabled(enabled)
        self.rename_action.setEnabled(enabled)
        if hasattr(self, "add_menu_bar"):
            self.add_menu_bar.setEnabled(enabled)

    def _refresh_recent(self) -> None:
        self.recent_list.clear()
        from PySide6.QtCore import QSettings
        settings = QSettings("Puki3D", "Puki3D")
        recent = settings.value("recent_projects", [], type=list)
        valid = []
        for entry in recent:
            path = Path(entry)
            if path.is_file() and path.suffix == PROJECT_EXTENSION:
                try:
                    project = Project.load(path)
                except ProjectError:
                    continue
                item = QListWidgetItem(f"▱   {project.name}     ·     {PROJECT_TYPES[project.kind]}\n      {path}")
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                self.recent_list.addItem(item)
                valid.append(str(path))
        settings.setValue("recent_projects", valid[:8])
        if not valid:
            empty = QListWidgetItem("Aún no hay proyectos recientes. Crea uno para empezar.")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            empty.setForeground(QColor(MUTED))
            self.recent_list.addItem(empty)

    def _remember_project(self, project: Project) -> None:
        from PySide6.QtCore import QSettings
        settings = QSettings("Puki3D", "Puki3D")
        recent = settings.value("recent_projects", [], type=list)
        settings.setValue("recent_projects", [str(project.path), *[p for p in recent if p != str(project.path)]][:8])
        self._refresh_recent()

    def _open_recent_item(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self._open_project(Path(path))

    def create_project(self) -> None:
        dialog = CreateProjectDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            project = Project.create(projects_root(), dialog.name.text(), dialog.kind.currentData())
        except ProjectError as exc:
            QMessageBox.warning(self, "No se pudo crear el proyecto", str(exc))
            return
        self._remember_project(project)
        self.show_project(project)

    def open_project_dialog(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "Abrir proyecto Puki3D", str(projects_root()), f"Proyectos Puki3D (*{PROJECT_EXTENSION})")
        if selected:
            self._open_project(Path(selected))

    def _open_project(self, path: Path) -> None:
        try:
            project = Project.load(path)
        except ProjectError as exc:
            QMessageBox.warning(self, "No se pudo abrir el proyecto", str(exc))
            return
        self._remember_project(project)
        self.show_project(project)

    def show_project(self, project: Project) -> None:
        self.project = project
        self.project_title.setText(project.name)
        self.project_type.setText(PROJECT_TYPES[project.kind])
        self.save_state.setText("●  Guardado")
        self.setWindowTitle(f"{project.name} — Puki3D")
        self._refresh_outliner(None)
        self._update_transform_fields(None)
        self._set_project_actions(True)
        self.stack.setCurrentWidget(self.workspace)

    def save_project(self) -> None:
        if not self.project:
            return
        self.autosave_timer.stop()
        try:
            self.project.save()
        except ProjectError as exc:
            QMessageBox.critical(self, "No se pudo guardar", str(exc))
            return
        self.save_state.setText("●  Guardado")
        self._remember_project(self.project)

    def rename_project(self) -> None:
        if not self.project:
            return
        from PySide6.QtWidgets import QInputDialog
        name, accepted = QInputDialog.getText(self, "Renombrar proyecto", "Nuevo nombre", text=self.project.name)
        if not accepted:
            return
        try:
            self.project.rename(name)
        except ProjectError as exc:
            QMessageBox.warning(self, "No se pudo renombrar", str(exc))
            return
        self._remember_project(self.project)
        self.show_project(self.project)

    def show_home(self) -> None:
        if self.project:
            self.autosave_timer.stop()
            try:
                self.project.save()
            except ProjectError as exc:
                QMessageBox.critical(self, "No se pudo guardar", str(exc))
                return
        self.project = None
        self.setWindowTitle("Puki3D")
        self._set_project_actions(False)
        self._refresh_recent()
        self.stack.setCurrentWidget(self.home)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt event API
        if self.project:
            self.autosave_timer.stop()
            try:
                self.project.save()
            except ProjectError as exc:
                QMessageBox.critical(self, "No se pudo guardar", str(exc))
                event.ignore()
                return
        event.accept()


def run_app() -> int:
    surface_format = QSurfaceFormat()
    surface_format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    surface_format.setVersion(2, 1)
    surface_format.setProfile(QSurfaceFormat.OpenGLContextProfile.NoProfile)
    surface_format.setDepthBufferSize(24)
    surface_format.setSamples(4)
    QSurfaceFormat.setDefaultFormat(surface_format)
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Puki3D")
    app.setOrganizationName("Puki3D")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()
