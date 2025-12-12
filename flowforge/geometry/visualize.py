from typing import Tuple
import os
import cv2  # type: ignore
from .primitives import GeometryOutput


def _shape_color(shape_type: str) -> Tuple[int, int, int]:
    """
    BGR colors for visibility on most images.
    """
    mapping = {
        "process": (0, 200, 0),      # green
        "decision": (200, 120, 0),   # blue-ish (BGR order -> orange-ish)
        "terminator": (0, 0, 220),   # red
        "input_output": (220, 0, 220),
        "connector": (200, 200, 0),
        "unknown": (0, 220, 220),    # yellow
    }
    return mapping.get(shape_type, (0, 220, 220))


def draw_geometry(image_path: str, geometry: GeometryOutput, output_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Draw shapes (bboxes + label + optional text)
    for shape in geometry.shapes:
        x1, y1, x2, y2 = shape.bbox
        color = _shape_color(shape.shape_type)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{shape.id}:{shape.shape_type}"
        cv2.putText(
            img,
            label,
            (int(x1), max(0, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
        if shape.text:
            text = shape.text
            text = (text[:40] + "…") if len(text) > 40 else text
            cv2.putText(
                img,
                text,
                (int(x1) + 4, int(y1) + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (50, 50, 50),
                2,
                lineType=cv2.LINE_AA,
            )

    # Draw connectors (lines/arrows)
    offset_row = 0
    for conn in geometry.connectors:
        poly_color = (255, 0, 0)
        text = f"{conn.id}:{conn.from_id}->{conn.to_id}"
        if conn.points and len(conn.points) >= 2:
            start = (int(conn.points[0][0]), int(conn.points[0][1]))
            end = (int(conn.points[-1][0]), int(conn.points[-1][1]))
            cv2.arrowedLine(
                img,
                start,
                end,
                poly_color,
                2,
                tipLength=0.2,
                line_type=cv2.LINE_AA,
            )
            label_pos = (start[0] + 5, max(0, start[1] - 5))
        else:
            label_pos = (10, 30 + 18 * offset_row)
            offset_row += 1
        if conn.label:
            text += f" [{conn.label}]"
        cv2.putText(
            img,
            text,
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            poly_color,
            1,
            lineType=cv2.LINE_AA,
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    return output_path


