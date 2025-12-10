from flowforge.graph.builder import parse_semantic_json


def test_parse_semantic_json_minimal():
    raw = """
    {
      "layout": {"orientation": "top-down", "swimlanes": false, "estimated_rows": 1},
      "nodes": [
        {
          "id": "n0",
          "approx_position": {"row": 0, "col": 0},
          "inferred_shape": "terminator",
          "text_summary": "Start",
          "role": "start"
        },
        {
          "id": "n1",
          "approx_position": {"row": 1, "col": 0},
          "inferred_shape": "process",
          "text_summary": "Do thing"
        }
      ],
      "edges": [
        {"from": "n0", "to": "n1", "label": null, "possible_labels": null}
      ],
      "notes": []
    }
    """
    skel = parse_semantic_json(raw)
    assert skel.layout["orientation"] == "top-down"
    assert len(skel.nodes) == 2
    assert len(skel.edges) == 1
    assert skel.edges[0].from_id == "n0"
    assert skel.edges[0].to_id == "n1"


