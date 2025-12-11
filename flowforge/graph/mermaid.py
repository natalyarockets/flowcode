from .schema import SemanticSkeleton


def to_mermaid(skel: SemanticSkeleton) -> str:
    """
    Convert SemanticSkeleton to a Mermaid flowchart string.
    Shape mapping:
    - process: [text]
    - decision: {text}
    - terminator: ([text])
    - input_output: [/text/]
    - unknown/other: [text]
    """
    def label_for(shape: str, text: str) -> str:
        safe = (text or "").replace('"', "'")
        if shape == "decision":
            return f"{{{safe}}}"
        if shape == "terminator":
            return f"([{safe}])"
        if shape == "input_output":
            return f"[/{safe}/]"
        return f"[{safe}]"

    lines = ["flowchart TD" if skel.layout.get("orientation") != "left-right" else "flowchart LR"]
    # Nodes
    for n in skel.nodes:
        lines.append(f'    {n.id}{label_for(n.inferred_shape, n.text_summary)}')
    # Edges
    for e in skel.edges:
        if e.to_id is None:
            continue
        if e.label:
            lines.append(f'    {e.from_id} --|{e.label}|--> {e.to_id}')
        else:
            lines.append(f'    {e.from_id} --> {e.to_id}')
    return "\n".join(lines)

