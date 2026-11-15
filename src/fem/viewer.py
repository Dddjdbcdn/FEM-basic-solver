"""A small GGUI postprocessor. Rendering never changes the solved displacement."""

from pathlib import Path
from time import perf_counter

import numpy as np
import taichi as ti

from .elasticity import init_runtime
from .input import surface_faces

# A sequential blue-cyan-yellow-red map, shared by surface colors and legend.
_RAMP = np.array(
    [
        [0.12, 0.20, 0.66],
        [0.08, 0.55, 0.87],
        [0.12, 0.80, 0.76],
        [0.83, 0.90, 0.32],
        [0.99, 0.59, 0.12],
        [0.80, 0.12, 0.16],
    ]
)

_PART_COLORS = np.array(
    [
        [0.40, 0.65, 1.0],
        [1.0, 0.65, 0.25],
        [0.30, 0.85, 0.65],
        [0.85, 0.50, 0.90],
        [1.0, 0.85, 0.35],
        [0.40, 0.85, 0.95],
    ]
)


def color_map(values, low, high):
    normalized = np.zeros_like(values)
    if high > low:
        normalized = (values - low) / (high - low)
    knots = np.linspace(0, 1, len(_RAMP))
    return np.ascontiguousarray(
        np.stack([np.interp(normalized, knots, _RAMP[:, i]) for i in range(3)], axis=-1),
        dtype=np.float32,
    )


def _vector_field(data):
    data = np.ascontiguousarray(data, dtype=np.float32)
    field = ti.Vector.field(data.shape[1], ti.f32, shape=len(data))
    field.from_numpy(data)
    return field


def _face_normals(vertices):
    triangles = vertices.reshape(-1, 3, 3)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return np.repeat(normals, 3, axis=0)


def _color_range(values):
    low, high = float(values.min()), float(values.max())
    # Avoid amplifying numerical noise into a full rainbow on a uniform field.
    if high - low <= 1e-8 * max(abs(low), abs(high)):
        low, high = min(low, 0.0), max(high, 0.0)
    return low, high


def _force_arrows(points, forces):
    lines = []
    for point, force in zip(points, forces):
        direction = force / np.linalg.norm(force)
        side = np.cross(direction, [0, 0, 1])
        if np.linalg.norm(side) == 0:
            side = np.cross(direction, [0, 1, 0])
        side /= np.linalg.norm(side)
        tip = point + 0.08 * direction
        wing1 = tip - 0.02 * direction + 0.01 * side
        wing2 = tip - 0.02 * direction - 0.01 * side
        lines.extend([point, tip, tip, wing1, tip, wing2])
    return np.array(lines)


def _legend():
    vertices, values = [], []
    for i in range(100):
        t0, t1 = i / 100, (i + 1) / 100
        y0, y1 = 0.14 + 0.51 * t0, 0.14 + 0.51 * t1
        vertices.extend(
            [(0.935, y0), (0.958, y0), (0.958, y1), (0.935, y0), (0.958, y1), (0.935, y1)]
        )
        values.extend([t0, t0, t1, t0, t1, t1])
    return _vector_field(np.array(vertices)), _vector_field(color_map(np.array(values), 0, 1))


def view(
    mesh,
    result=None,
    boundary=None,
    title="Solid FEM",
    screenshot=None,
    frames=None,
    up="z",
):
    """Show results, or inspect a mesh. A screenshot renders once without a window."""
    init_runtime()
    faces, owners = surface_faces(mesh)
    face_nodes = faces.ravel()
    length = float(np.max(np.ptp(mesh.points, axis=0)))
    center = (mesh.points.min(axis=0) + mesh.points.max(axis=0)) / 2
    rest = (mesh.points - center) / length
    displacement = np.zeros_like(rest)
    if result is not None:
        displacement = result.displacement / length
    vertices = _vector_field(rest[face_nodes])
    normals = _vector_field(_face_normals(rest[face_nodes]))
    colors = _vector_field(np.full((len(face_nodes), 3), 0.65))
    part_colors = np.full((len(mesh.tetra), 3), 0.65)
    for index, elements in enumerate(mesh.parts.values()):
        part_colors[elements] = _PART_COLORS[index % len(_PART_COLORS)]
    show_parts = result is None and bool(mesh.parts)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0).ravel()
    wire = _vector_field(rest[edges])
    fixed_ids = np.array([], dtype=int)
    loaded_ids = np.array([], dtype=int)
    if boundary is not None:
        fixed_ids = np.flatnonzero(np.any(boundary.fixed, axis=1))
        loaded_ids = np.flatnonzero(np.linalg.norm(boundary.force, axis=1) > 0)
    support = _vector_field(rest[fixed_ids]) if len(fixed_ids) else None
    arrow = None
    if len(loaded_ids):
        arrow = _vector_field(_force_arrows(rest[loaded_ids], boundary.force[loaded_ids]))
    legend_vertices, legend_colors = _legend()
    axis_origin = np.array([0.05, -0.35, -0.65])
    axis_positions = np.repeat(axis_origin[None, :], 6, axis=0)
    axis_positions[[1, 3, 5]] += np.eye(3) * 0.12
    axis_field = _vector_field(axis_positions)
    axis_colors = _vector_field(
        np.repeat([[0.95, 0.30, 0.28], [0.28, 0.85, 0.44], [0.32, 0.60, 1]], 2, axis=0)
    )
    max_u = float(np.max(np.linalg.norm(displacement, axis=1)))
    default_scale = min(0.08 / max_u, 1000.0) if max_u > 0 else 1.0
    scale, max_scale = default_scale, max(2 * default_scale, 1.0)
    default_distance = 2.25 if np.sort(np.ptp(rest, axis=0))[1] > 0.3 else 1.65
    yaw, pitch, distance = 1.05, 0.42, default_distance
    field_index, averaged, show_wire, show_bc = 0, True, False, True
    last_cursor = None
    mouse_was_down = False
    orbit_drag = False
    last_state = None
    names = [
        "Von Mises stress [Pa]",
        "Displacement magnitude [m]",
        "Normal stress XX [Pa]",
        "Normal stress YY [Pa]",
        "Normal stress ZZ [Pa]",
    ]
    window = ti.ui.Window(title, (1280, 800), show_window=screenshot is None, fps_limit=60)
    canvas, scene, gui = window.get_canvas(), window.get_scene(), window.get_gui()
    if screenshot is not None:
        # GGUI initializes ImGui only after its first draw. Warm up before the saved frame.
        window.get_image_buffer_as_numpy()
    camera = ti.ui.Camera()
    vertical = "xyz".index(up)
    horizontal = [axis for axis in range(3) if axis != vertical]
    camera.up(*np.eye(3)[vertical])
    camera.fov(45)
    frame, last_time = 0, perf_counter()
    while window.running:
        now = perf_counter()
        dt = min(now - last_time, 0.1)
        last_time = now
        events = window.get_events(ti.ui.PRESS) if screenshot is None else []
        for event in events:
            if event.key == ti.ui.ESCAPE:
                window.running = False
            elif event.key == "w":
                show_wire = not show_wire
            elif event.key == "a":
                averaged = not averaged
            elif event.key == "r":
                yaw, pitch, distance, scale = 1.05, 0.42, default_distance, default_scale
            elif event.key in ("1", "2"):
                field_index = int(event.key) - 1
        cursor = np.array(window.get_cursor_pos()) if screenshot is None else np.zeros(2)
        if screenshot is None:
            mouse_down = window.is_pressed(ti.ui.LMB, ti.ui.RMB)
            if mouse_down and not mouse_was_down:
                # Start orbits on the scene, so dragging a GUI slider doesn't rotate it.
                x, y = cursor[0], 1.0 - cursor[1]
                over_controls = 0.02 <= x <= 0.285 and 0.025 <= y <= 0.615
                over_legend = result is not None and x >= 0.79
                orbit_drag = not (over_controls or over_legend)
            elif not mouse_down:
                orbit_drag = False
            if orbit_drag and mouse_was_down and last_cursor is not None:
                delta = cursor - last_cursor
                yaw -= 4 * delta[0]
                pitch = float(np.clip(pitch + 4 * delta[1], -1.45, 1.45))
            mouse_was_down = mouse_down
        last_cursor = cursor
        if screenshot is None and window.is_pressed("=", "+"):
            distance = max(0.65, distance * np.exp(-dt))
        if screenshot is None and window.is_pressed("-"):
            distance = min(5.0, distance * np.exp(dt))
        with gui.sub_window("SOLID FEM", 0.02, 0.025, 0.265, 0.59):
            gui.text("3D / STATIC / LINEAR ELASTIC", color=(0.35, 0.78, 0.95))
            gui.text(f"{len(mesh.points):,} nodes | {len(mesh.tetra):,} Tet4")
            gui.text(" ")
            if result is not None:
                gui.text(names[field_index])
                if gui.button("Next result"):
                    field_index = (field_index + 1) % len(names)
                averaged = gui.checkbox("Average stress tensors at nodes", averaged)
                if field_index == 1:
                    gui.text("Nodal displacement")
                elif averaged:
                    gui.text("Averaged stress view")
                else:
                    gui.text("Raw element stress view")
                scale = gui.slider_float("Deformation", scale, 0.0, max_scale)
                gui.text(f"Displayed deformation: {scale:.1f}x")
                gui.text(f"Max |u|: {max_u * length:.4e} m")
                gui.text(f"Max element VM: {result.von_mises.max():.4e} Pa")
                gui.text(f"Residual: {result.relative_residual:.2e}")
            else:
                gui.text("Mesh inspection")
            if mesh.parts:
                show_parts = gui.checkbox("Color by part", show_parts)
                if show_parts:
                    for index, name in enumerate(mesh.parts):
                        gui.text(name, color=tuple(_PART_COLORS[index % len(_PART_COLORS)]))
            show_wire = gui.checkbox("Wireframe", show_wire)
            show_bc = gui.checkbox("Supports and force arrows", show_bc)
            distance = gui.slider_float("Camera distance", distance, 0.65, 5.0)
            if gui.button("Reset view"):
                yaw, pitch, distance, scale = 1.05, 0.42, default_distance, default_scale
            gui.text(" ")
            gui.text("Left/right drag: rotate | +/-: zoom")
            gui.text("W: edges | A: averaging | R: reset")
            gui.text("1: stress | 2: displacement | Esc: exit")
            gui.text("Axes: X red / Y green / Z blue")
            if boundary is not None:
                gui.text("Supports green / Forces pink")
        state = (scale, field_index, averaged, show_parts)
        if state != last_state:
            deformed = rest + scale * displacement
            vertices.from_numpy(np.ascontiguousarray(deformed[face_nodes], dtype=np.float32))
            normals.from_numpy(
                np.ascontiguousarray(_face_normals(deformed[face_nodes]), dtype=np.float32)
            )
            wire.from_numpy(np.ascontiguousarray(deformed[edges], dtype=np.float32))
            if support is not None:
                support.from_numpy(np.ascontiguousarray(deformed[fixed_ids], dtype=np.float32))
            if arrow is not None:
                arrow.from_numpy(
                    np.ascontiguousarray(
                        _force_arrows(deformed[loaded_ids], boundary.force[loaded_ids]),
                        dtype=np.float32,
                    )
                )
            if show_parts:
                colors.from_numpy(np.repeat(part_colors[owners], 3, axis=0).astype(np.float32))
            elif result is not None:
                if field_index == 1:
                    values = np.linalg.norm(result.displacement, axis=1)
                    surface_values = values[face_nodes]
                else:
                    if field_index == 0:
                        values = result.nodal_von_mises if averaged else result.von_mises
                    else:
                        stress = result.nodal_stress if averaged else result.stress
                        values = stress[:, field_index - 2]
                    surface_values = (
                        values[face_nodes] if averaged else np.repeat(values[owners], 3)
                    )
                # Fix the range over the entire volume, including interior elements.
                low, high = _color_range(values)
                colors.from_numpy(color_map(surface_values, low, high))
            else:
                colors.from_numpy(np.full((len(face_nodes), 3), 0.65, dtype=np.float32))
            last_state = state
        offset = np.zeros(3)
        offset[horizontal[0]] = distance * np.cos(pitch) * np.cos(yaw)
        offset[horizontal[1]] = distance * np.cos(pitch) * np.sin(yaw)
        offset[vertical] = distance * np.sin(pitch)
        # Leave room for the controls on the left, regardless of the up axis.
        right = np.cross(-offset, np.eye(3)[vertical])
        target = -0.12 * right / np.linalg.norm(right)
        camera.position(*(target + offset))
        camera.lookat(*target)
        scene.set_camera(camera)
        scene.ambient_light((0.88, 0.88, 0.88))
        scene.point_light(pos=(1.5, 2, 3), color=(0.18, 0.18, 0.18))
        scene.mesh(vertices, normals=normals, per_vertex_color=colors, two_sided=True)
        if show_wire:
            scene.lines(wire, width=1.0, color=(0.08, 0.11, 0.15))
        if show_bc:
            if support is not None:
                scene.particles(support, radius=0.004, color=(0.25, 0.92, 0.55))
            if arrow is not None:
                scene.lines(arrow, width=2, color=(1.0, 0.38, 0.65))
        scene.lines(axis_field, width=3, per_vertex_color=axis_colors)
        canvas.set_background_color((0.045, 0.065, 0.095))
        canvas.scene(scene)
        if result is not None and not show_parts:
            canvas.triangles(legend_vertices, per_vertex_color=legend_colors)
            with gui.sub_window("RESULT RANGE", 0.79, 0.025, 0.19, 0.13):
                gui.text(names[field_index])
                gui.text("Whole-volume range")
            # Labels sit beside the canvas ramp, outside any covering panel.
            for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                value = low + fraction * (high - low)
                with gui.sub_window(
                    f"{fraction:.2f}", 0.82, 0.33 + 0.51 * (1 - fraction), 0.11, 0.065
                ):
                    gui.text(f"{value:.3e}")
        frame += 1
        if screenshot is not None:
            screenshot = Path(screenshot)
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            window.save_image(str(screenshot))
            break
        window.show()
        if frames is not None and frame >= frames:
            break
    window.destroy()
