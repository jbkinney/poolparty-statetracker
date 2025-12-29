import json
from pathlib import Path
from typing import Union, Optional, Dict, List
from .pool import Pool

def visualize(
    pool: Pool,
    output_html: str = "computation_graph.html",
    sample_count: int = 16,
    max_samples_per_node: int = 1000,
    pool_colors: Optional[Dict[str, str]] = None,
    show_design_cards: bool = False,
    track_pools: Optional[List[str]] = None,
) -> str:
    """
    Generates an interactive HTML dashboard for exploring the Pool computation graph.

    This visualization includes:
    1.  **Topology Graph**: A node-link diagram showing the lineage and operations (concatenation, mutation, scan, etc.) that form the library.
    2.  **Linear Sequence View**: A track-based, coordinate-aligned view of the sequences, allowing inspection of spatial relationships (e.g., insertion sites, mutation distributions).
    3.  **Inspector Panel**: Detailed metadata and sampled sequences for any selected node.
    4.  **Design Cards Panel** (optional): A table view of per-sequence metadata for all tracked pools.

    The output is a standalone HTML file containing all necessary data and scripts (D3.js).

    Args:
        pool (Pool): The root Pool object representing the library to visualize.
        output_html (str): File path where the generated HTML will be saved. Defaults to "computation_graph.html".
        sample_count (int): Number of random sequences to sample from the pool for visualization and statistics. Defaults to 16.
        max_samples_per_node (int): Maximum number of sampled sequences to store per node in the output file. Trimming prevents excessive file size for deep graphs. Defaults to 1000.
        pool_colors (Optional[Dict[str, str]]): A dictionary mapping `pool.name` to CSS color strings (e.g., {"Promoter": "#ff0000"}). Used to customize node and track colors.
        show_design_cards (bool): If True, includes a Design Cards panel showing per-sequence metadata for all tracked pools. Defaults to False.
        track_pools (Optional[List[str]]): List of pool names to track in design cards. If None, tracks all named pools. Only used when show_design_cards=True.

    Returns:
        str: The path to the generated HTML file.

    Raises:
        ValueError: If `sample_count` or `max_samples_per_node` is not positive.
    """

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if max_samples_per_node <= 0:
        raise ValueError("max_samples_per_node must be positive")

    # Global Sequence Registry to compress JSON
    class SequenceRegistry:
        def __init__(self):
            self.map = {}
            self.list = []
        def get_id(self, seq):
            s = str(seq) if seq is not None else ""
            if s not in self.map:
                self.map[s] = len(self.list)
                self.list.append(s)
            return self.map[s]
        def to_json(self):
            return json.dumps(self.list)

    registry = SequenceRegistry()

    # Generate sequences with computation graph and optionally design cards
    result = pool.generate_seqs(
        num_seqs=sample_count,
        return_computation_graph=True,
        return_design_cards=show_design_cards,
        track_pools=track_pools,
    )
    graph_dict = result["graph"]
    node_sequences = result["node_sequences"]
    
    # Process design cards if requested
    design_cards_data = {"columns": [], "rows": [], "has_pool_columns": False}
    if show_design_cards and "design_cards" in result:
        dc = result["design_cards"]
        design_cards_data["columns"] = dc.keys
        # Convert rows to list of lists for JSON
        for i in range(len(dc)):
            row = dc.get_row(i)
            design_cards_data["rows"].append([row.get(k) for k in dc.keys])
        # Check if there are pool-specific columns (beyond sequence_id and sequence_length)
        design_cards_data["has_pool_columns"] = len(dc.keys) > 2

    nodes = {}
    referenced_ids = set()
    pool_count = 0
    literal_count = 0

    for raw_node in graph_dict["nodes"]:
        node_id = str(raw_node["node_id"])
        parent_ids = [str(pid) for pid in raw_node.get("parent_ids", [])]
        referenced_ids.update(parent_ids)

        if raw_node.get("type") == "Pool":
            pool_count += 1
            op_name = (raw_node.get("op") or "").strip()
            
            # Extract pool name and determine if composite
            pool_name = raw_node.get("name")
            is_composite = op_name in ('+', '*', 'slice')
            
            # Build label: prefer name, then op, then "Pool"
            if pool_name:
                label = pool_name
                if op_name:
                    label = f"{pool_name} ({op_name})"
            elif op_name:
                label = op_name
            else:
                label = "Pool"
            
            kind = "pool-composite" if is_composite else "pool"
            mode = raw_node.get("mode") or ""
            num_states_raw = raw_node.get("num_states")
            if isinstance(num_states_raw, str):
                num_states_display = num_states_raw
            elif num_states_raw is None:
                num_states_display = ""
            else:
                num_states_display = str(num_states_raw)
            literal_value = None
            pool_name_val = pool_name
            is_composite_val = is_composite
        else:
            literal_count += 1
            value_type = raw_node.get("value_type", "literal")
            value = raw_node.get("value")
            label = str(value) if value is not None else value_type
            kind = f"literal:{value_type}"
            mode = value_type
            num_states_display = "1"
            literal_value = str(value) if value is not None else None

        short_label = label if len(label) <= 20 else f"{label[:17]}..."

        raw_samples = node_sequences.get(node_id, [])
        if isinstance(raw_samples, list):
            cleaned_samples = ["" if s is None else str(s) for s in raw_samples]
        elif raw_samples is None:
            cleaned_samples = []
        else:
            cleaned_samples = [str(raw_samples)]

        sequence_total = len(cleaned_samples)
        
        # Store IDs instead of strings for sample list
        sample_ids = [registry.get_id(s) for s in cleaned_samples[:max_samples_per_node]]

        nodes[node_id] = {
            "id": node_id,
            "label": short_label,
            "title": label,
            "kind": kind,
            "mode": mode,
            "num_states": num_states_display,
            "literal": literal_value,
            "parent_ids": parent_ids,
            "sample_ids": sample_ids, # Compressed
            "sequence_total": sequence_total,
            "pool_name": pool_name_val,
            "is_composite": is_composite_val,
        }

    consumer_counts = {node_id: 0 for node_id in nodes}
    for node in nodes.values():
        for parent_id in node["parent_ids"]:
            if parent_id in consumer_counts:
                consumer_counts[parent_id] += 1

    for node_id, node in nodes.items():
        node["parent_count"] = len(node["parent_ids"])
        node["consumer_count"] = consumer_counts.get(node_id, 0)

    root_candidates = [node_id for node_id in nodes if node_id not in referenced_ids]
    if not root_candidates:
        root_candidates = [str(graph_dict["nodes"][0]["node_id"])]

    def build_subtree(node_id, path_ids, path_label):
        node = nodes[node_id]
        children = []
        for idx, parent_id in enumerate(node["parent_ids"]):
            if parent_id not in nodes or parent_id in path_ids:
                continue
            child_label = f"{path_label}.{idx}"
            children.append(build_subtree(parent_id, path_ids | {parent_id}, child_label))
        return {
            "id": node["id"],
            "uid": path_label,
            "label": node["label"],
            "title": node["title"],
            "kind": node["kind"],
            "mode": node["mode"],
            "numStates": node["num_states"],
            "literal": node["literal"],
            "parentCount": node["parent_count"],
            "consumerCount": node["consumer_count"],
            "sample_ids": node["sample_ids"],
            "sequence_total": node["sequence_total"],
            "pool_name": node.get("pool_name"),
            "is_composite": node.get("is_composite", False),
            "children": children,
        }
    
    # --- Linear View Logic ---
    node_map = {n["node_id"]: n for n in graph_dict["nodes"]}
    
    SCAN_OPERATIONS = {
        'insertion_scan', 'deletion_scan', 'shuffle_scan', 
        'di_shuffle_scan', 'subseq'
    }
    
    def get_masked_consensus(samples: list) -> str:
        if not samples:
            return ""
        str_samples = [str(s) for s in samples]
        ref_len = len(str_samples[0])
        for s in str_samples:
            if len(s) != ref_len:
                return str_samples[0]
        consensus = []
        for i in range(ref_len):
            char = str_samples[0][i]
            is_constant = True
            for s in str_samples[1:]:
                if s[i] != char:
                    is_constant = False
                    break
            consensus.append(char if is_constant else '.')
        return "".join(consensus)

    def build_linear_hierarchy(node_id, start_offset=0, is_variable_position=False):
        node_info = node_map[node_id]
        seqs = node_sequences.get(str(node_id), [])
        all_seqs = []
        if isinstance(seqs, list):
            all_seqs = [str(s) for s in seqs]
        elif seqs is not None:
            all_seqs = [str(seqs)]
            
        current_seq = all_seqs[0] if all_seqs else ""
        consensus_seq = get_masked_consensus(all_seqs) if all_seqs else ""
        
        length = len(current_seq)
        
        # Store IDs instead of full strings
        all_seq_ids = [registry.get_id(s) for s in all_seqs]
        
        hierarchy = {
            "id": str(node_id),
            "type": node_info.get("type", "Pool"),
            "start": start_offset,
            "end": start_offset + length,
            "length": length,
            "label": nodes[str(node_id)]["title"],
            "pool_name": nodes[str(node_id)]["pool_name"], 
            "seq": current_seq,
            "consensus_seq": consensus_seq,
            "all_seq_ids": all_seq_ids,
            "variable_position": is_variable_position,
            "overlap_children": False,
            "children": []
        }
        
        op = node_info.get("op")
        parent_ids = node_info.get("parent_ids", [])
        is_scan_op = op in SCAN_OPERATIONS
        
        if node_info.get("type") == "Pool":
            if op == "+":
                current_pos = start_offset
                for pid in parent_ids:
                    child = build_linear_hierarchy(pid, current_pos, False)
                    hierarchy["children"].append(child)
                    current_pos += child["length"]
            elif op == "*":
                pool_parent_id = None
                multiplier = 1
                for pid in parent_ids:
                    pnode = node_map[pid]
                    if pnode.get("type") == "Pool":
                        pool_parent_id = pid
                    elif pnode.get("type") == "literal" and pnode.get("value_type") == "int":
                        multiplier = int(pnode.get("value"))
                
                if pool_parent_id is not None and multiplier > 0:
                    current_pos = start_offset
                    child_seqs = node_sequences.get(str(pool_parent_id), [])
                    child_seq = child_seqs[0] if isinstance(child_seqs, list) and child_seqs else str(child_seqs)
                    child_len = len(child_seq)
                    for i in range(multiplier):
                        child = build_linear_hierarchy(pool_parent_id, current_pos, False)
                        child["label"] = f"{child['label']} [{i+1}]" 
                        hierarchy["children"].append(child)
                        current_pos += child_len
            elif op == 'slice':
                 if parent_ids:
                     child = build_linear_hierarchy(parent_ids[0], start_offset, False)
                     hierarchy["children"].append(child)
            elif len(parent_ids) == 1:
                pid = parent_ids[0]
                child = build_linear_hierarchy(pid, start_offset, False)
                hierarchy["children"].append(child)
            elif len(parent_ids) > 1:
                hierarchy["overlap_children"] = True
                for pid in parent_ids:
                    pnode = node_map[pid]
                    if pnode.get("type") == "Pool":
                        child = build_linear_hierarchy(pid, start_offset, is_scan_op)
                        hierarchy["children"].append(child)
        return hierarchy

    if len(root_candidates) == 1:
        tree_data = build_subtree(root_candidates[0], {root_candidates[0]}, root_candidates[0])
        root_int = int(root_candidates[0])
        linear_data = build_linear_hierarchy(root_int)
    else:
        tree_data = {
            "id": "root",
            "uid": "root",
            "label": "Outputs",
            "title": "Graph Outputs",
            "kind": "root",
            "mode": "",
            "numStates": str(len(root_candidates)),
            "literal": None,
            "parentCount": 0,
            "consumerCount": len(root_candidates),
            "sample_ids": [],
            "sequence_total": 0,
            "children": [
                build_subtree(node_id, {node_id}, node_id)
                for node_id in root_candidates
            ],
        }
        
        if root_candidates:
            linear_data = build_linear_hierarchy(int(root_candidates[0]))
        else:
            linear_data = {}

    def compute_depth(node):
        if not node.get("children"):
            return 1
        return 1 + max(compute_depth(child) for child in node["children"])

    stats = {
        "total_nodes": len(nodes),
        "pool_nodes": pool_count,
        "literal_nodes": literal_count,
        "root_count": len(root_candidates),
        "max_depth": compute_depth(tree_data),
        "unique_sequences": len(registry.list)
    }

    tree_json = json.dumps(tree_data, indent=2)
    stats_json = json.dumps(stats, indent=2)
    linear_json = json.dumps(linear_data, indent=2)
    colors_json = json.dumps(pool_colors or {}, indent=2)
    registry_json = registry.to_json()
    design_cards_json = json.dumps(design_cards_data, indent=2)
    show_design_cards_json = "true" if show_design_cards else "false"

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>PoolParty Designer View</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
    /* Modern Palette (Slate + Indigo) */
    --bg-app: #f8fafc;
    --bg-panel: #ffffff;
    --border-subtle: #e2e8f0;
    --border-strong: #cbd5e1;
    
    --text-primary: #0f172a;
    --text-secondary: #64748b;
    --text-tertiary: #94a3b8;
    
    --primary: #4f46e5;       /* Indigo 600 */
    --primary-hover: #4338ca; /* Indigo 700 */
    --primary-light: #eef2ff; /* Indigo 50 */
    
    --accent: #0ea5e9;        /* Sky 500 */
    --danger: #ef4444;        /* Red 500 */
    --success: #10b981;       /* Emerald 500 */
    --warning: #f59e0b;       /* Amber 500 */
    
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    
    --radius-sm: 0.375rem;
    --radius-md: 0.5rem;
    --radius-lg: 0.75rem;
}

* { box-sizing: border-box; }

body {
    margin: 0;
    background: var(--bg-app);
    color: var(--text-primary);
    font-family: "Inter", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
    height: 100vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}

/* Header */
header {
    height: 60px;
    background: #1e293b; /* Slate 800 */
    color: white;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    box-shadow: var(--shadow-md);
    z-index: 20;
    flex-shrink: 0;
}
.header-left { display: flex; align-items: center; gap: 16px; }
.header-right { display: flex; align-items: center; gap: 20px; }

h1 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #f8fafc;
}
.stats-badge {
    background: rgba(255,255,255,0.1);
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 12px;
    color: #cbd5e1;
    font-weight: 500;
}

/* Toggle Switches */
.toggle-switch {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    font-size: 13px; 
    color: #cbd5e1;
    user-select: none;
    transition: color 0.2s;
}
.toggle-switch:hover { color: white; }
.toggle-switch input {
    appearance: none;
    width: 36px;
    height: 20px;
    background: #475569;
    border-radius: 99px;
    position: relative;
    transition: background 0.2s;
    cursor: pointer;
}
.toggle-switch input::after {
    content: '';
    position: absolute;
    top: 2px; left: 2px;
    width: 16px; height: 16px;
    background: white;
    border-radius: 50%;
    transition: transform 0.2s cubic-bezier(0.4, 0.0, 0.2, 1);
}
.toggle-switch input:checked { background: var(--primary); }
.toggle-switch input:checked::after { transform: translateX(16px); }

/* Main Layout */
main {
    flex: 1;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    overflow: hidden;
}

/* Panels */
.panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    transition: box-shadow 0.2s, flex 0.3s ease;
}
.panel:focus-within { box-shadow: var(--shadow-md); border-color: var(--border-strong); }
.panel.hidden { display: none !important; }

.panel-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #f8fafc;
}
.panel-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-title svg { width: 16px; height: 16px; color: var(--text-tertiary); }

.panel-toolbar {
    display: flex;
    gap: 8px;
    align-items: center;
}

.panel-content {
    flex: 1;
    min-height: 0;
    position: relative;
    display: flex;
    flex-direction: column;
}

/* Splitter Sections */
#top-section {
    flex: 3;
    display: flex;
    min-height: 200px;
    overflow: hidden;
}
#bottom-section {
    flex: 2;
    min-height: 150px;
    display: flex;
    flex-direction: column;
}

/* Controls */
button.btn {
    appearance: none;
    border: 1px solid transparent;
    background: white;
    color: var(--text-secondary);
    border-color: var(--border-subtle);
    padding: 6px 12px;
    border-radius: var(--radius-sm);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
button.btn:hover {
    background: #f1f5f9;
    color: var(--text-primary);
    border-color: var(--border-strong);
}
button.btn:active { transform: translateY(1px); }

button.btn-primary {
    background: var(--primary);
    color: white;
    border-color: transparent;
    box-shadow: 0 1px 2px rgba(79, 70, 229, 0.3);
}
button.btn-primary:hover { background: var(--primary-hover); color: white; }

button.btn-icon {
    padding: 6px;
    color: var(--text-tertiary);
    background: transparent;
    border: none;
    box-shadow: none;
}
button.btn-icon:hover { background: #f1f5f9; color: var(--danger); }

.control-group {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 8px;
    background: white;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
}
.control-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

input[type="range"] {
    -webkit-appearance: none;
    width: 80px;
    height: 4px;
    background: var(--border-subtle);
    border-radius: 2px;
    outline: none;
}
input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px; height: 14px;
    background: white;
    border: 2px solid var(--primary);
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    transition: transform 0.1s;
}
input[type="range"]::-webkit-slider-thumb:hover { transform: scale(1.1); }

/* Graph Specifics */
#graph-panel {
    min-width: 300px;
}
#graph-wrapper {
    flex: 1;
    position: relative;
    background-image: radial-gradient(#cbd5e1 1px, transparent 1px);
    background-size: 20px 20px;
    background-color: #ffffff;
}

/* Node Styles */
.node rect {
    fill: white;
    stroke: var(--border-strong);
    stroke-width: 1.5px;
    rx: 8px; ry: 8px; /* Pill shape mostly */
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.03));
}
.node:hover rect {
    filter: drop-shadow(0 4px 8px rgba(0,0,0,0.08));
    stroke: var(--text-secondary);
    transform: translateY(-1px);
}
.node.selected rect {
    stroke: var(--primary);
    stroke-width: 2.5px;
    filter: drop-shadow(0 0 0 4px var(--primary-light));
}

/* Node Color Coding */
.node.pool rect { stroke: var(--primary); fill: #f5f7ff; }
.node.pool-composite rect { stroke: #8b5cf6; fill: #fbf7ff; }
.node.literal rect { stroke: var(--warning); fill: #fffbeb; }
.node.root rect { stroke: var(--success); fill: #f0fdf4; }

.node text.title {
    font-family: "Inter", sans-serif;
    font-size: 12px;
    font-weight: 600;
    fill: var(--text-primary);
    pointer-events: none;
}
.node text.subtitle {
    font-family: "Inter", sans-serif;
    font-size: 10px;
    fill: var(--text-secondary);
    pointer-events: none;
}

.link {
    fill: none;
    stroke: #cbd5e1;
    stroke-width: 2px;
    transition: stroke 0.3s;
}

/* Linear View */
#linear-view-wrapper {
    background: #ffffff;
    flex: 1;
    overflow: hidden;
    position: relative;
}
#linear-view {
    width: 100%;
    height: 100%;
}
.track-rect {
    stroke: white;
    stroke-width: 1px;
    rx: 4px;
    transition: opacity 0.2s;
    cursor: pointer;
}
.track-rect:hover { opacity: 0.85; }
.track-rect.variable-position {
    stroke-dasharray: 4 2;
    stroke-width: 2px;
    stroke: rgba(0,0,0,0.3);
}
.track-label {
    font-size: 11px;
    font-weight: 600;
    fill: white;
    pointer-events: none;
    text-shadow: 0 1px 3px rgba(0,0,0,0.8);
}
.track-seq {
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
    fill: white;
    text-shadow: 0 1px 3px rgba(0,0,0,0.8);
}
#linear-panel {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
}
.axis text {
    fill: var(--text-tertiary);
    font-family: "Inter", sans-serif;
    font-size: 10px;
}
.axis path, .axis line { stroke: var(--border-subtle); }

/* Details Panel */
#detail-panel {
    width: 320px;
    min-width: 200px;
    max-width: calc(100vw - 400px);
    flex-shrink: 0;
    border-left: 1px solid var(--border-subtle);
    display: none; /* flex when active */
}
.detail-scroll { padding: 20px; overflow-y: auto; flex: 1; }

.prop-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.prop-item {
    background: #f8fafc;
    padding: 10px;
    border-radius: var(--radius-md);
    border: 1px solid var(--border-subtle);
}
.prop-label { font-size: 10px; color: var(--text-tertiary); text-transform: uppercase; margin-bottom: 4px; font-weight: 600; }
.prop-value { font-size: 13px; color: var(--text-primary); font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

.seq-list {
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    height: 300px;
    overflow-y: auto;
    position: relative;
    background: #fff;
}
.seq-row {
    display: flex;
    height: 32px;
    align-items: center;
    padding: 0 12px;
    border-bottom: 1px solid #f1f5f9;
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
}
.seq-row:nth-child(even) { background: #f8fafc; }
.seq-idx { width: 40px; color: var(--text-tertiary); user-select: none; flex-shrink: 0; margin-right: 8px; text-align: right; }

/* Design Cards Panel */
#design-cards-panel {
    width: 400px;
    min-width: 200px;
    max-width: calc(100vw - 400px);
    flex-shrink: 0;
    border-left: 1px solid var(--border-subtle);
    display: none;
}
.dc-scroll { padding: 16px; padding-top: 8px; overflow: hidden; flex: 1; display: flex; flex-direction: column; }
.dc-toolbar {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 12px;
}
.dc-search-wrapper {
    flex: 1;
    position: relative;
}
.dc-search {
    width: 100%;
    padding: 8px 12px 8px 32px;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    font-size: 12px;
    font-family: inherit;
    background: white;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.dc-search:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px var(--primary-light);
}
.dc-search-icon {
    position: absolute;
    left: 10px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--text-tertiary);
    pointer-events: none;
}
.dc-search-clear {
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    background: none;
    border: none;
    color: var(--text-tertiary);
    cursor: pointer;
    padding: 2px 6px;
    font-size: 14px;
    border-radius: 50%;
    display: none;
}
.dc-search-clear:hover { background: #f1f5f9; color: var(--text-primary); }
.dc-filter-count {
    font-size: 11px;
    color: var(--text-secondary);
    white-space: nowrap;
}
.dc-header-container {
    position: relative;
    overflow: hidden;
    flex-shrink: 0;
    border: 1px solid var(--border-subtle);
    border-bottom: none;
    border-radius: var(--radius-md) var(--radius-md) 0 0;
    background: #f8fafc;
}
.dc-header-scroll {
    overflow-x: hidden;
    overflow-y: hidden;
}
.dc-header-row {
    display: flex;
    position: relative;
    border-bottom: 2px solid var(--border-strong);
    padding-right: 150px;
}
.dc-header-cell {
    flex-shrink: 0;
    position: relative;
    min-width: 55px;
    box-sizing: border-box;
}
.dc-header-cell .th-inner {
    position: absolute;
    bottom: 8px;
    left: 8px;
    transform: rotate(-45deg);
    transform-origin: left bottom;
    white-space: nowrap;
    font-weight: 600;
    font-size: 10px;
    color: var(--text-secondary);
    letter-spacing: 0.02em;
}
.dc-table-wrapper {
    flex: 1;
    overflow-x: auto;
    overflow-y: auto;
    border: 1px solid var(--border-subtle);
    border-top: none;
    border-radius: 0 0 var(--radius-md) var(--radius-md);
    background: #fff;
    position: relative;
}
.dc-table-phantom {
    position: absolute;
    width: 100%;
    pointer-events: none;
}
.dc-table-content {
    position: absolute;
    width: 100%;
}
.dc-table {
    border-collapse: separate;
    border-spacing: 0;
    font-size: 12px;
    margin-right: 150px;
    width: 100%;
}
.dc-table tbody td {
    padding: 6px 12px;
    border-bottom: 1px solid #f1f5f9;
    font-family: "JetBrains Mono", monospace;
    font-size: 11px;
    white-space: nowrap;
    color: var(--text-primary);
    min-width: 55px;
    box-sizing: border-box;
}
.dc-table tr:nth-child(even) td { background: #f8fafc; }
.dc-table tr:hover td { background: #eef2ff; }
.dc-table td.dc-null { color: var(--text-tertiary); font-style: italic; }
.dc-table td.dc-highlight { background: #fef3c7 !important; }
.dc-empty-msg {
    padding: 20px;
    text-align: center;
    color: var(--text-tertiary);
    font-size: 12px;
}
.dc-no-results {
    padding: 40px 20px;
    text-align: center;
    color: var(--text-tertiary);
    font-size: 13px;
}

/* Tooltip */
#tooltip {
    position: absolute;
    background: rgba(15, 23, 42, 0.95);
    backdrop-filter: blur(4px);
    color: white;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 12px;
    box-shadow: var(--shadow-lg);
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s;
    z-index: 100;
    max-width: 300px;
    border: 1px solid rgba(255,255,255,0.1);
}
.tooltip-header { font-weight: 600; font-size: 13px; margin-bottom: 8px; color: white; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 6px;}
.tooltip-row { display: flex; justify-content: space-between; margin-bottom: 4px; gap: 16px; }
.tooltip-row span:first-child { color: #94a3b8; }

/* Splitters */
.splitter { background: var(--border-subtle); z-index: 10; transition: background 0.2s; flex-shrink: 0; }
.splitter:hover, .splitter.dragging { background: var(--primary); }
.splitter-h { width: 6px; cursor: col-resize; }
.splitter-v { height: 6px; cursor: row-resize; width: 100%; }

</style>
</head>
<body>

<header>
    <div class="header-left">
        <h1>PoolParty <span style="font-weight:300; opacity:0.7;">Designer</span></h1>
        <div id="stats-display" class="stats-badge">Loading...</div>
    </div>
    <div class="header-right">
        <label class="toggle-switch">
            <input type="checkbox" checked id="toggle-graph"> Graph
        </label>
        <label class="toggle-switch">
            <input type="checkbox" checked id="toggle-linear"> Linear View
        </label>
        <label class="toggle-switch">
            <input type="checkbox" id="toggle-details"> Details
        </label>
        <label class="toggle-switch" id="toggle-dc-label" style="display:none">
            <input type="checkbox" id="toggle-design-cards"> Design Cards
        </label>
    </div>
</header>

<main>
    <div id="top-section">
        <div class="panel" id="graph-panel" style="flex:1">
            <div class="panel-header">
                <div class="panel-title">
                    <!-- Icon -->
                    <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                    </svg>
                    Graph View
                </div>
                <div class="panel-toolbar">
                    <button class="btn" id="fit-btn">Fit View</button>
                    <div class="control-group">
                        <span class="control-label">Spacing</span>
                        <input type="range" id="spacing-x" min="80" max="300" value="140" title="Horizontal Spacing">
                        <input type="range" id="spacing-y" min="60" max="300" value="140" title="Vertical Spacing">
            </div>
                    <button class="btn" id="expand-btn">Expand All</button>
                    <button class="btn" id="collapse-btn">Collapse</button>
                </div>
                </div>
            <div class="panel-content">
                <div id="graph-wrapper">
                    <svg id="graph-svg" width="100%" height="100%"></svg>
            </div>
                <!-- Floating Legend -->
                <div style="position:absolute; bottom:16px; left:16px; background:rgba(255,255,255,0.9); padding:8px 12px; border-radius:6px; box-shadow:var(--shadow-md); border:1px solid var(--border-subtle); font-size:11px; color:var(--text-secondary); display:flex; gap:12px; backdrop-filter:blur(4px);">
                    <div style="display:flex; align-items:center; gap:6px;"><span style="width:8px; height:8px; background:#4f46e5; border-radius:50%;"></span> Pool</div>
                    <div style="display:flex; align-items:center; gap:6px;"><span style="width:8px; height:8px; background:#8b5cf6; border-radius:50%;"></span> Composite</div>
                    <div style="display:flex; align-items:center; gap:6px;"><span style="width:8px; height:8px; background:#f59e0b; border-radius:50%;"></span> Literal</div>
                    <div style="display:flex; align-items:center; gap:6px;"><span style="width:8px; height:8px; background:#10b981; border-radius:50%;"></span> Root</div>
                </div>
            </div>
        </div>

        <div class="splitter splitter-h" id="split-d" style="display:none"></div>

        <div class="panel" id="detail-panel">
            <div class="panel-header">
                <div class="panel-title">
                    <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Node Details
                </div>
                <button class="btn btn-icon" id="close-detail">×</button>
            </div>
            <div class="detail-scroll">
                <h2 id="d-title" style="margin:0 0 4px; font-size:16px; font-weight:600;">-</h2>
                <div id="d-subtitle" style="color:var(--text-secondary); font-size:12px; font-family:'JetBrains Mono'; margin-bottom:20px;">-</div>
                
                <div class="prop-grid" id="d-props"></div>

                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <h3 style="margin:0; font-size:13px; font-weight:600;">Sequence List</h3>
                    <label style="font-size:11px; display:flex; align-items:center; gap:6px;">
                        <input type="checkbox" id="highlight-lower"> Mark Changes
                    </label>
                </div>
                
                <!-- Virtual List -->
                <div id="seq-list-container" class="seq-list">
                    <div id="seq-list-phantom" style="position:absolute; width:100%;"></div>
                    <div id="seq-list-content" style="position:absolute; width:100%;"></div>
                </div>
            </div>
        </div>

        <div class="splitter splitter-h" id="split-dc" style="display:none"></div>

        <div class="panel" id="design-cards-panel">
            <div class="panel-header">
                <div class="panel-title">
                    <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
                    </svg>
                    Design Cards
                </div>
                <button class="btn btn-icon" id="close-design-cards">×</button>
            </div>
            <div class="dc-scroll">
                <div class="dc-toolbar">
                    <div class="dc-search-wrapper">
                        <svg class="dc-search-icon" width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
                        </svg>
                        <input type="text" class="dc-search" id="dc-search" placeholder="Search values...">
                        <button class="dc-search-clear" id="dc-search-clear">×</button>
                    </div>
                    <span class="dc-filter-count" id="dc-filter-count"></span>
                </div>
                <div id="dc-info" style="margin-bottom:8px; font-size:12px; color:var(--text-secondary);"></div>
                <div class="dc-header-container">
                    <div class="dc-header-scroll" id="dc-header-scroll">
                        <div class="dc-header-row" id="dc-header-row"></div>
                    </div>
                </div>
                <div class="dc-table-wrapper" id="dc-table-wrapper">
                    <div class="dc-table-phantom" id="dc-table-phantom"></div>
                    <div class="dc-table-content" id="dc-table-content">
                        <table class="dc-table" id="dc-table">
                            <tbody id="dc-tbody"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="splitter splitter-v" id="split-v"></div>

    <div class="panel" id="linear-panel">
        <div class="panel-header">
            <div class="panel-title">
                <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h7" />
                </svg>
                Linear Sequence View
            </div>
            <div class="panel-toolbar">
                <div class="control-group">
                    <label style="display:flex; align-items:center; gap:6px; font-size:12px; color:var(--text-secondary); cursor:pointer;">
                        <input type="checkbox" id="uniform-layout"> Uniform Layout
                </label>
                </div>
                <div class="control-group">
                    <label style="display:flex; align-items:center; gap:6px; font-size:12px; color:var(--text-secondary); cursor:pointer;">
                        <input type="checkbox" id="show-text"> Text
                </label>
                </div>
                 <div class="control-group">
                    <label style="display:flex; align-items:center; gap:6px; font-size:12px; color:var(--text-secondary); cursor:pointer;">
                    <input type="checkbox" id="show-consensus"> Consensus
                </label>
            </div>
                <div class="control-group">
                    <label style="display:flex; align-items:center; gap:6px; font-size:12px; color:var(--text-secondary); cursor:pointer;">
                    <input type="checkbox" id="linear-highlight-lower"> Mark Changes
                </label>
            </div>
                <div class="control-group" id="seq-idx-control">
                    <span class="control-label">Seq ID</span>
                    <input type="number" id="seq-index" value="0" min="0" style="width:50px; border:none; background:transparent; font-family:inherit; font-size:12px; font-weight:600; text-align:right;">
        </div>
                <div class="control-group">
                    <span class="control-label">Zoom</span>
                    <input type="range" id="linear-zoom" min="0.001" max="50" step="0.001" value="1">
                </div>
                <button class="btn btn-primary" id="linear-fit">Fit</button>
            </div>
        </div>
        <div class="panel-content" id="linear-view-wrapper">
            <div id="linear-view"></div>
        </div>
    </div>
</main>

<div id="tooltip"></div>

<script>
// --- Data Injection ---
const data = __TREE_JSON__;
const stats = __STATS_JSON__;
const linearData = __LINEAR_JSON__;
const customColors = __COLORS_JSON__;
const seqRegistry = __REGISTRY_JSON__;
const designCardsData = __DESIGN_CARDS_JSON__;
const showDesignCards = __SHOW_DESIGN_CARDS__;

function getSeq(id) { return seqRegistry[id] || ""; }

// --- Utility: Debounce ---
function debounce(fn, delay) {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn.apply(this, args), delay);
    };
}

// --- Utility: RAF Throttle ---
function rafThrottle(fn) {
    let scheduled = false;
    return function(...args) {
        if (scheduled) return;
        scheduled = true;
        requestAnimationFrame(() => {
            fn.apply(this, args);
            scheduled = false;
        });
    };
}

// --- Stats Init ---
document.getElementById("stats-display").textContent = 
    `Nodes: ${stats.total_nodes} • Pools: ${stats.pool_nodes} • Literals: ${stats.literal_nodes} • Depth: ${stats.max_depth} • Roots: ${stats.root_count}`;

// --- D3 Graph Setup ---
const svg = d3.select("#graph-svg");
const gZoom = svg.append("g");
const gLinks = gZoom.append("g").attr("class", "links");
const gNodes = gZoom.append("g").attr("class", "nodes");

const zoom = d3.zoom()
    .scaleExtent([0.1, 4])
    .on("zoom", e => gZoom.attr("transform", e.transform));
svg.call(zoom);

// Global settings
let nodeSpacingX = 140, levelSpacing = 140;
const nodeW = 160, nodeH = 40;

const root = d3.hierarchy(data, d => d.children);
root.x0 = 0; root.y0 = 0;

// Render Initial
try {
    update(root);
    // Delay to ensure layout is computed for dimensions
    setTimeout(() => {
        try { fitGraph(); } catch(e) { console.error("Fit Graph Error:", e); }
        try { renderLinearView(linearData); } catch(e) { 
            console.error("Linear View Init Error:", e); 
            d3.select("#linear-view").html(`<div style="padding:20px;color:#ef4444;font-size:12px;">Error loading linear view: ${e.message}</div>`);
        }
    }, 300);
} catch(e) {
    console.error("Graph Init Error:", e);
}

// --- Graph Logic ---
function update(source) {
    const tree = d3.tree().nodeSize([nodeSpacingX, levelSpacing])
        .separation((a, b) => a.parent === b.parent ? 1.1 : 1.3);
    
    tree(root);
    
    const nodes = root.descendants();
    const links = root.links();
    
    // 1. Nodes
    const node = gNodes.selectAll("g.node")
        .data(nodes, d => d.data.uid || (d.data.uid = Math.random()));
    
    const nodeEnter = node.enter().append("g")
        .attr("class", d => `node ${categorise(d)}`)
        .attr("id", d => `node-${d.data.id}`)
        .attr("transform", d => `translate(${source.x0},${source.y0})`)
        .attr("cursor", "pointer")
        .on("click", handleClick)
        .on("dblclick", (e, d) => {
            if(d.children) { d._children = d.children; d.children = null; }
            else { d.children = d._children; d._children = null; }
            update(d);
        })
        .on("mouseenter", (e,d) => {
            showTooltip(e, d);
            highlightLinearNode(d.data.id, true); 
            highlightTreeNode(d.data.id, true);
        })
        .on("mouseleave", (e,d) => {
            hideTooltip();
            highlightLinearNode(null, false);
            unhighlightTreeNode(d.data.id);
        });

    nodeEnter.append("rect")
        .attr("width", nodeW).attr("height", nodeH)
        .attr("x", -nodeW/2).attr("y", -nodeH/2);

    nodeEnter.append("text").attr("class", "title")
        .attr("dy", "-4px").attr("text-anchor", "middle")
        .text(d => truncate(d.data.title || d.data.label, 20));

    nodeEnter.append("text").attr("class", "subtitle")
        .attr("dy", "10px").attr("text-anchor", "middle")
        .text(d => d.data.pool_name || d.data.mode || d.data.kind);

    const nodeUpdate = nodeEnter.merge(node);
    nodeUpdate.transition().duration(300)
        .attr("transform", d => `translate(${d.x},${d.y})`);
    
    node.exit().transition().duration(300)
        .attr("transform", d => `translate(${source.x},${source.y})`)
        .style("opacity", 0).remove();

    // 2. Links
    const link = gLinks.selectAll("path.link")
        .data(links, d => d.target.data.uid);
    
    link.enter().append("path").attr("class", "link")
        .attr("d", d => elbow(source, source))
        .merge(link).transition().duration(300)
        .attr("d", d => elbow(d.source, d.target));
        
    link.exit().transition().duration(300)
        .attr("d", d => elbow(source, source)).remove();

    nodes.forEach(d => { d.x0 = d.x; d.y0 = d.y; });
}

function elbow(s, t) {
    // Smooth bezier curve instead of sharp elbow
    return `M${s.x},${s.y} C${s.x},${(s.y + t.y)/2} ${t.x},${(s.y + t.y)/2} ${t.x},${t.y}`;
}

function categorise(d) {
    if (d.data.kind === 'root') return 'root';
    if (d.data.kind && d.data.kind.startsWith('literal')) return 'literal';
    if (d.data.kind === 'pool-composite') return 'pool-composite';
    return 'pool';
}

// --- Linear View Logic ---
let linearSvg, linearG, linearZoom;

function renderLinearView(rootData) {
    const container = d3.select("#linear-view");

    const cw = document.getElementById("linear-view-wrapper").clientWidth || 800;
    const ch = document.getElementById("linear-view-wrapper").clientHeight || 200;
    
    if (container.select("svg").empty()) {
        linearSvg = container.append("svg")
            .attr("width", "100%").attr("height", "100%");
            
        linearG = linearSvg.append("g");
        linearZoom = d3.zoom().scaleExtent([0.001, 10]).on("zoom", e => linearG.attr("transform", e.transform));
        linearSvg.call(linearZoom);
        linearSvg.call(linearZoom.transform, d3.zoomIdentity.translate(20, 40));
    }
    
    linearSvg.attr("viewBox", [0, 0, cw, ch]);
    
    // Clear content
    linearG.html("");

    if (!rootData) {
        linearG.append("text").attr("x", 20).attr("y", 30).style("fill","#888").text("No sequence data.");
        return;
    }

    const uniform = d3.select("#uniform-layout").property("checked");
    const showText = d3.select("#show-text").property("checked");
    const showConsensus = d3.select("#show-consensus").property("checked");
    const seqIndex = +d3.select("#seq-index").property("value") || 0;
    const zoomLvl = +d3.select("#linear-zoom").property("value") || 1;
    
    d3.select("#seq-idx-control").style("display", showConsensus ? "none" : "flex");
    
    let layout = JSON.parse(JSON.stringify(rootData));
    
    // Pre-process sequences
    function prep(node) {
        if (showConsensus) {
            if (node.consensus_seq) node.seq = node.consensus_seq;
        } else if (node.all_seq_ids && node.all_seq_ids.length > 0) {
            node.seq = getSeq(node.all_seq_ids[seqIndex % node.all_seq_ids.length]);
        }
        // Always calc actual length
        node.actualLength = node.seq ? node.seq.length : (node.length || 1);
        if(node.children) node.children.forEach(prep);
    }
    prep(layout);

    // Always calc actual positions
    function calcActual(n, start) {
        n.actualStart = start; 
        n.actualEnd = start + n.actualLength;
        let cPos = start;
        if (n.children) {
            if (n.overlap_children) n.children.forEach(c => calcActual(c, start));
            else n.children.forEach(c => { calcActual(c, cPos); cPos += c.actualLength; });
        }
    }
    calcActual(layout, 0);

    if (uniform) {
        function calcUniform(n) {
            if (!n.children || !n.children.length) { n.uLen = 50; return 50; }
            n.uLen = n.children.reduce((s, c) => s + calcUniform(c), 0);
            return n.uLen;
        }
        calcUniform(layout);
        function applyUniform(n, start) {
            n.start = start; n.end = start + n.uLen;
            let cPos = start;
            if (n.children) n.children.forEach(c => { applyUniform(c, cPos); cPos += c.uLen; });
        }
        applyUniform(layout, 0);
    } else {
        function useActual(n) {
            n.start = n.actualStart; n.end = n.actualEnd; n.length = n.actualLength;
            if(n.children) n.children.forEach(useActual);
        }
        useActual(layout);
    }

    const totalLen = layout.end - layout.start;
    const baseW = cw - 60;

    // Adapt max zoom to sequence length (ensure text is readable)
    const desiredMaxZoom = (totalLen * 12) / Math.max(100, baseW);
    d3.select("#linear-zoom").attr("max", Math.max(50, desiredMaxZoom));

    const scaleX = d3.scaleLinear().domain([-0.5, totalLen - 0.5]).range([0, Math.max(1, baseW * zoomLvl)]);
    const rowH = 28;  
    const gap = 8;

    // Draw Axis
    if (!uniform) {
        const totalWidth = Math.max(1, baseW * zoomLvl);
        const ticks = scaleX.ticks(Math.max(2, totalWidth / 80)).filter(t => Number.isInteger(t) && t >= 0);
        const axis = d3.axisTop(scaleX).tickValues(ticks).tickFormat(d3.format("d"));
        linearG.append("g").attr("class", "axis").attr("transform", "translate(0,-10)").call(axis);
    }

    // Traverse & Draw
    const colorScale = d3.scaleOrdinal(d3.schemeTableau10);
    
    function draw(node, depth) {
        if (node.type !== 'literal') {
            const y = depth * (rowH + gap);
            const x0 = scaleX(node.start - 0.5);
            const x1 = scaleX(node.end - 0.5);
            const w = Math.max(2, x1 - x0);
            
            const g = linearG.append("g").attr("transform", `translate(${x0}, ${y})`);
            
            const color = (customColors && node.pool_name && customColors[node.pool_name]) 
                ? customColors[node.pool_name] 
                : colorScale(node.label);
            
            const clipId = `clip-${node.id}`;
            g.append("defs").append("clipPath")
                .attr("id", clipId)
                .append("rect")
                .attr("width", w)
                .attr("height", rowH);

            const rect = g.append("rect")
                .attr("class", "track-rect")
                .classed("variable-position", node.variable_position)
                .attr("width", w).attr("height", rowH)
                .attr("fill", color)
                .attr("id", `linear-${node.id}`)
                .on("click", () => {
                    highlightTreeNode(node.id);
                    const d = d3.select(`#node-${node.id}`).datum();
                    if(d) showDetail(d);
                })
                .on("mouseenter", (e) => {
                    highlightTreeNode(node.id, true);
                    showTooltip(e, { 
                        data: { 
                            title: node.label, 
                            kind: node.pool_name || "Pool",
                            position: `${node.actualStart}-${node.actualEnd}`,
                            length: node.actualEnd - node.actualStart
                        }
                    });
            })
            .on("mouseleave", () => {
                    unhighlightTreeNode(node.id);
                hideTooltip();
                });

            // Text
            const showHighlightLower = d3.select("#linear-highlight-lower").property("checked");
            
            if (showText && node.seq && w/node.seq.length > 5) {
                const charW = w / node.seq.length;
                        const coords = [];
                for (let j = 0; j < node.seq.length; j++) {
                            coords.push(j * charW + charW / 2);
                        }

                const textEl = g.append("text")
                            .attr("class", "track-seq")
                    .attr("y", rowH/2)
                    .attr("dominant-baseline", "middle")
                    .attr("text-anchor", "middle")
                    .attr("clip-path", `url(#${clipId})`);
                            
                        if (showHighlightLower) {
                     node.seq.split("").forEach((char, idx) => {
                                const tspan = textEl.append("tspan")
                                    .attr("x", coords[idx])
                                    .text(char);
                                if (char >= 'a' && char <= 'z') {
                            tspan.style("fill", "#ef4444").style("font-weight", "bold").style("text-shadow", "none");
                                }
                            });
                        } else {
                     textEl.attr("x", coords.join(" ")).text(node.seq);
                }
            } else if (w > 20) {
                g.append("text")
                        .attr("class", "track-label")
                    .attr("x", w/2).attr("y", rowH/2).attr("dy", "3px")
                        .attr("text-anchor", "middle")
                        .attr("clip-path", `url(#${clipId})`)
                    .text(truncate(node.label, Math.floor(w / 7)));
            }
        }
        if (node.children) node.children.forEach(c => draw(c, depth + 1));
    }
    draw(layout, 0);
}

// --- Interaction Helpers ---
function handleClick(e, d) {
    e.stopPropagation();
    showDetail(d);
    highlightTreeNode(d.data.id);
}

function highlightTreeNode(id, hover) {
    if (hover) {
        d3.select(`#node-${id} rect`)
            .style("stroke", "#ef4444")
            .style("stroke-width", "3px")
            .style("filter", "drop-shadow(0 0 4px rgba(239, 68, 68, 0.5))");
            } else {
        d3.selectAll(".node").classed("selected", false);
        if(id) d3.select(`#node-${id}`).classed("selected", true);
    }
}

function unhighlightTreeNode(id) {
    d3.select(`#node-${id} rect`)
        .style("stroke", null)
        .style("stroke-width", null)
        .style("filter", null);
}

function highlightLinearNode(id, hover) {
    d3.selectAll(".track-rect").style("opacity", 0.4);
    if(id) d3.select(`#linear-${id}`).style("opacity", 1).style("stroke", "#000");
    else d3.selectAll(".track-rect").style("opacity", 1).style("stroke", null);
}

// --- Details Panel ---
let currentDetail = null;
const ROW_H = 32;

function showDetail(d) {
    currentDetail = d;
    const p = d3.select("#detail-panel");
    p.style("display", "flex");
    d3.select("#split-d").style("display", "block");
    d3.select("#toggle-details").property("checked", true);
    
    d3.select("#d-title").text(d.data.title || d.data.label);
    d3.select("#d-subtitle").text(`${d.data.kind} • ID: ${d.data.id}`);
    
    const props = [
        ["Op Mode", d.data.mode],
        ["States", d.data.numStates],
        ["Parents", d.data.parentCount],
        ["Pool Name", d.data.pool_name],
        ["Total Generated", d.data.sequence_total]
    ];
    
    d3.select("#d-props").html(props.map(([k,v]) => 
        v ? `<div class="prop-item"><div class="prop-label">${k}</div><div class="prop-value" title="${v}">${v}</div></div>` : ""
    ).join(""));
    
    renderList();
}

function renderList() {
    if (!currentDetail) return;
    try {
        const list = document.getElementById("seq-list-content");
        if(!list) return; 

        const ids = currentDetail.data.sample_ids || [];
    const phantom = document.getElementById("seq-list-phantom");
        if(phantom) phantom.style.height = (ids.length * ROW_H) + "px";
        
        const container = document.getElementById("seq-list-container");
        if(!container) return;

        const start = Math.floor(container.scrollTop / ROW_H);
        const count = Math.ceil(container.clientHeight / ROW_H) + 2;
    
    let html = "";
    const hl = d3.select("#highlight-lower").property("checked");
        
        for(let i = start; i < Math.min(ids.length, start + count); i++) {
            let seq = getSeq(ids[i]);
            if(hl && seq) {
                seq = seq.replace(/[a-z]/g, m => `<span style="color:var(--danger); font-weight:bold">${m}</span>`);
            }
            html += `<div class="seq-row"><div class="seq-idx">${i}</div>${seq}</div>`;
        }
        list.innerHTML = html;
        list.style.transform = `translateY(${start * ROW_H}px)`;
    } catch(e) {
        console.error("Render List Error:", e);
    }
}

document.getElementById("seq-list-container").addEventListener("scroll", renderList);

// --- Tooltip ---
function showTooltip(e, d) {
    const t = d3.select("#tooltip");
    const content = `
        <div class="tooltip-header">${d.data.title || d.data.label}</div>
        <div class="tooltip-row"><span>Type</span> <span>${d.data.kind}</span></div>
        ${d.data.pool_name ? `<div class="tooltip-row"><span>Name</span> <span>${d.data.pool_name}</span></div>` : ""}
        ${d.data.mode ? `<div class="tooltip-row"><span>Mode</span> <span>${d.data.mode}</span></div>` : ""}
        ${d.data.position ? `<div class="tooltip-row"><span>Position</span> <span>${d.data.position}</span></div>` : ""}
        ${d.data.length ? `<div class="tooltip-row"><span>Length</span> <span>${d.data.length}</span></div>` : ""}
        ${d.data.numStates ? `<div class="tooltip-row"><span>States</span> <span>${d.data.numStates}</span></div>` : ""}
    `;
    
    // Edge detection for tooltip
    const pageX = e.pageX, pageY = e.pageY;
    const w = window.innerWidth, h = window.innerHeight;
    let left = pageX + 15, top = pageY + 15;
    
    if (left + 200 > w) left = pageX - 220;
    if (top + 150 > h) top = pageY - 160;

    t.html(content)
     .style("opacity", 1)
     .style("left", left + "px")
     .style("top", top + "px");
}
function hideTooltip() { d3.select("#tooltip").style("opacity", 0); }

// --- Controls ---
d3.select("#fit-btn").on("click", fitGraph);
d3.select("#linear-fit").on("click", () => {
    if(!linearSvg) return;
    const bb = linearG.node().getBBox();
    if(bb.width === 0) return;
    const w = document.getElementById("linear-view-wrapper").clientWidth;
    const h = document.getElementById("linear-view-wrapper").clientHeight;
    const k = Math.min(w/bb.width, h/bb.height) * 0.9;
    linearSvg.transition().duration(500).call(
        linearZoom.transform, 
        d3.zoomIdentity.translate(w/2 - (bb.x + bb.width/2)*k, h/2 - (bb.y + bb.height/2)*k).scale(k)
    );
});

// Debounced graph spacing updates
const debouncedGraphUpdate = debounce(() => update(root), 50);
d3.select("#spacing-x").on("input", function() { nodeSpacingX = +this.value; debouncedGraphUpdate(); });
d3.select("#spacing-y").on("input", function() { levelSpacing = +this.value; debouncedGraphUpdate(); });

d3.select("#toggle-graph").on("change", function() { 
    d3.select("#graph-panel").classed("hidden", !this.checked);
});
d3.select("#toggle-linear").on("change", function() { 
    d3.select("#linear-panel").classed("hidden", !this.checked);
});
d3.select("#toggle-details").on("change", function() {
    const p = d3.select("#detail-panel");
    if(this.checked) { p.style("display", "flex"); d3.select("#split-d").style("display", "block"); }
    else { p.style("display", "none"); d3.select("#split-d").style("display", "none"); }
});
d3.select("#close-detail").on("click", () => {
    d3.select("#detail-panel").style("display", "none");
    d3.select("#split-d").style("display", "none");
    d3.select("#toggle-details").property("checked", false);
});

// Redraws with debouncing for continuous inputs
const redrawLinear = () => renderLinearView(linearData);
const debouncedRedrawLinear = debounce(redrawLinear, 30);
d3.select("#uniform-layout").on("change", redrawLinear);
d3.select("#show-text").on("change", redrawLinear);
d3.select("#show-consensus").on("change", redrawLinear);
d3.select("#linear-zoom").on("input", debouncedRedrawLinear);
d3.select("#seq-index").on("input", debouncedRedrawLinear);
d3.select("#linear-highlight-lower").on("change", redrawLinear);
d3.select("#highlight-lower").on("change", renderList);

function fitGraph() {
    const b = gZoom.node().getBBox();
    if (b.width === 0) return;
    const w = document.getElementById("graph-wrapper").clientWidth;
    const h = document.getElementById("graph-wrapper").clientHeight;
    const s = Math.min(w / (b.width + 100), h / (b.height + 100), 1.5);
    svg.transition().duration(750).call(
        zoom.transform, 
        d3.zoomIdentity.translate(w/2 - (b.x + b.width/2)*s, h/2 - (b.y + b.height/2)*s).scale(s)
    );
}
function truncate(s, n) { return (s && s.length > n) ? s.slice(0, n) + "…" : s; }

// Splitter Logic
const splitD = document.getElementById("split-d");
const splitV = document.getElementById("split-v");
let isDragD = false, isDragV = false;

// Minimum widths for panels
const MIN_GRAPH_WIDTH = 300;
const MIN_PANEL_WIDTH = 200;
const MAIN_PADDING = 32; // main element padding (16px * 2)

splitD.addEventListener("mousedown", () => isDragD = true);
splitV.addEventListener("mousedown", () => isDragV = true);

function getAvailableWidth() {
    return document.getElementById("top-section").offsetWidth;
}

document.addEventListener("mousemove", e => {
    if(isDragD) {
        e.preventDefault();
        const topSection = document.getElementById("top-section");
        const topRect = topSection.getBoundingClientRect();
        const dcPanel = document.getElementById("design-cards-panel");
        const dcVisible = dcPanel.style.display !== 'none';
        const dcWidth = dcVisible ? dcPanel.offsetWidth + 6 : 0; // 6 for splitter
        
        // Calculate position relative to top-section
        const mouseX = e.clientX - topRect.left;
        const availableW = topRect.width;
        
        // Detail panel width = available width - mouseX - design cards panel
        const detailWidth = availableW - mouseX - dcWidth;
        
        // Clamp to valid range
        const maxDetailWidth = availableW - MIN_GRAPH_WIDTH - dcWidth;
        const clampedWidth = Math.max(MIN_PANEL_WIDTH, Math.min(detailWidth, maxDetailWidth));
        
        document.getElementById("detail-panel").style.width = clampedWidth + "px";
    }
    if(isDragV) {
        e.preventDefault();
        const topH = e.pageY - 60; // header offset
        if(topH > 100 && topH < window.innerHeight - 100) {
            document.getElementById("top-section").style.flex = "none";
            document.getElementById("top-section").style.height = topH + "px";
        }
    }
});
document.addEventListener("mouseup", () => {
    if(isDragD || isDragV || isDragDC) { isDragD = false; isDragV = false; isDragDC = false; setTimeout(() => { fitGraph(); redrawLinear(); }, 100); }
});

// --- Design Cards Panel with Virtual Scrolling & Search ---
const splitDC = document.getElementById("split-dc");
let isDragDC = false;

// Design Cards State
const dcState = {
    cols: [],
    allRows: [],
    filteredRows: [],
    filteredIndices: [],
    searchTerm: '',
    cellWidth: 70,
    rowHeight: 28,
    headerHeight: 100,
    initialized: false
};

if (showDesignCards) {
    document.getElementById("toggle-dc-label").style.display = "flex";
    initDesignCards();
}

function initDesignCards() {
    if (!designCardsData.columns || designCardsData.columns.length === 0) {
        document.getElementById("dc-info").textContent = "No design card data available.";
        return;
    }
    
    dcState.cols = designCardsData.columns;
    dcState.allRows = designCardsData.rows;
    dcState.filteredRows = dcState.allRows;
    dcState.filteredIndices = dcState.allRows.map((_, i) => i);
    
    // Info line
    const info = document.getElementById("dc-info");
    if (!designCardsData.has_pool_columns) {
        info.innerHTML = `<span style="color:var(--warning);">⚠</span> No named pools tracked. Only sequence metadata shown.`;
    } else {
        info.textContent = `${dcState.allRows.length} sequences × ${dcState.cols.length} columns`;
    }
    
    // Calculate header height
    const maxColLength = Math.max(...dcState.cols.map(c => c.length));
    dcState.headerHeight = Math.max(100, Math.ceil(maxColLength * 6 * 0.707) + 20);
    document.querySelector(".dc-header-container").style.height = dcState.headerHeight + "px";
    
    // Build header
    const headerRow = document.getElementById("dc-header-row");
    let headerHtml = "";
    for (const col of dcState.cols) {
        headerHtml += `<div class="dc-header-cell" style="width:${dcState.cellWidth}px; height:${dcState.headerHeight}px;" title="${col}"><span class="th-inner">${col}</span></div>`;
    }
    headerRow.innerHTML = headerHtml;
    
    // Set phantom height for virtual scrolling
    updateDCPhantomHeight();
    
    // Render initial visible rows
    renderDCVisibleRows();
    
    // Setup scroll handler with RAF throttling
    const tableWrapper = document.getElementById("dc-table-wrapper");
    const headerScroll = document.getElementById("dc-header-scroll");
    
    const handleDCScroll = rafThrottle(() => {
        headerScroll.scrollLeft = tableWrapper.scrollLeft;
        renderDCVisibleRows();
    });
    tableWrapper.addEventListener("scroll", handleDCScroll);
    
    // Setup search with debouncing
    const searchInput = document.getElementById("dc-search");
    const searchClear = document.getElementById("dc-search-clear");
    const filterCount = document.getElementById("dc-filter-count");
    
    const handleSearch = debounce((term) => {
        dcState.searchTerm = term.toLowerCase().trim();
        filterDCRows();
        updateDCPhantomHeight();
        renderDCVisibleRows();
        
        // Update filter count
        if (dcState.searchTerm) {
            filterCount.textContent = `${dcState.filteredRows.length} / ${dcState.allRows.length}`;
            searchClear.style.display = "block";
        } else {
            filterCount.textContent = "";
            searchClear.style.display = "none";
        }
    }, 150);
    
    searchInput.addEventListener("input", (e) => handleSearch(e.target.value));
    
    searchClear.addEventListener("click", () => {
        searchInput.value = "";
        handleSearch("");
        searchInput.focus();
    });
    
    dcState.initialized = true;
}

function filterDCRows() {
    if (!dcState.searchTerm) {
        dcState.filteredRows = dcState.allRows;
        dcState.filteredIndices = dcState.allRows.map((_, i) => i);
        return;
    }
    
    dcState.filteredRows = [];
    dcState.filteredIndices = [];
    
    for (let i = 0; i < dcState.allRows.length; i++) {
        const row = dcState.allRows[i];
        let match = false;
        for (const val of row) {
            if (val !== null && val !== undefined) {
                const strVal = typeof val === "object" ? JSON.stringify(val) : String(val);
                if (strVal.toLowerCase().includes(dcState.searchTerm)) {
                    match = true;
                    break;
                }
            }
        }
        if (match) {
            dcState.filteredRows.push(row);
            dcState.filteredIndices.push(i);
        }
    }
}

function updateDCPhantomHeight() {
    const phantom = document.getElementById("dc-table-phantom");
    phantom.style.height = (dcState.filteredRows.length * dcState.rowHeight) + "px";
}

function renderDCVisibleRows() {
    const tableWrapper = document.getElementById("dc-table-wrapper");
    const tbody = document.getElementById("dc-tbody");
    const content = document.getElementById("dc-table-content");
    
    if (!dcState.filteredRows.length) {
        tbody.innerHTML = "";
        content.style.transform = "translateY(0)";
        if (dcState.searchTerm) {
            tbody.innerHTML = `<tr><td colspan="${dcState.cols.length}" class="dc-no-results">No matching rows found</td></tr>`;
        }
        return;
    }
    
    const scrollTop = tableWrapper.scrollTop;
    const viewHeight = tableWrapper.clientHeight;
    const buffer = 5; // Extra rows above/below viewport
    
    const startIdx = Math.max(0, Math.floor(scrollTop / dcState.rowHeight) - buffer);
    const endIdx = Math.min(dcState.filteredRows.length, Math.ceil((scrollTop + viewHeight) / dcState.rowHeight) + buffer);
    
    let html = "";
    for (let i = startIdx; i < endIdx; i++) {
        const row = dcState.filteredRows[i];
        const origIdx = dcState.filteredIndices[i];
        const evenClass = origIdx % 2 === 0 ? "" : ""; // We'll handle with CSS nth-child on visible
        
        html += "<tr>";
        for (let j = 0; j < row.length; j++) {
            const val = row[j];
            let cellClass = "";
            let content = "";
            let title = "";
            
            if (val === null || val === undefined) {
                cellClass = "dc-null";
                content = "—";
            } else if (typeof val === "object") {
                content = JSON.stringify(val);
                title = content;
                if (content.length > 8) content = content.slice(0, 6) + "..";
            } else {
                const strVal = String(val);
                title = strVal;
                content = strVal.length > 8 ? strVal.slice(0, 6) + ".." : strVal;
                
                // Highlight matching cells
                if (dcState.searchTerm && strVal.toLowerCase().includes(dcState.searchTerm)) {
                    cellClass = "dc-highlight";
                }
            }
            
            html += `<td class="${cellClass}" style="min-width:${dcState.cellWidth}px; max-width:${dcState.cellWidth}px;" title="${title}">${content}</td>`;
        }
        html += "</tr>";
    }
    
    tbody.innerHTML = html;
    content.style.transform = `translateY(${startIdx * dcState.rowHeight}px)`;
}

// Design Cards Toggle
d3.select("#toggle-design-cards").on("change", function() {
    const p = d3.select("#design-cards-panel");
    if(this.checked) { 
        p.style("display", "flex"); 
        d3.select("#split-dc").style("display", "block"); 
    } else { 
        p.style("display", "none"); 
        d3.select("#split-dc").style("display", "none"); 
    }
});

d3.select("#close-design-cards").on("click", () => {
    d3.select("#design-cards-panel").style("display", "none");
    d3.select("#split-dc").style("display", "none");
    d3.select("#toggle-design-cards").property("checked", false);
});

// Design Cards Splitter
splitDC.addEventListener("mousedown", () => isDragDC = true);

document.addEventListener("mousemove", e => {
    if(isDragDC) {
        e.preventDefault();
        const topSection = document.getElementById("top-section");
        const topRect = topSection.getBoundingClientRect();
        const detailPanel = document.getElementById("detail-panel");
        const detailVisible = detailPanel.style.display !== 'none';
        const detailWidth = detailVisible ? detailPanel.offsetWidth + 6 : 0; // 6 for splitter
        
        // Calculate position relative to top-section
        const mouseX = e.clientX - topRect.left;
        const availableW = topRect.width;
        
        // Design cards panel width = available width - mouseX
        const dcWidth = availableW - mouseX;
        
        // Clamp to valid range
        const maxDCWidth = availableW - MIN_GRAPH_WIDTH - detailWidth;
        const clampedWidth = Math.max(MIN_PANEL_WIDTH, Math.min(dcWidth, maxDCWidth));
        
        document.getElementById("design-cards-panel").style.width = clampedWidth + "px";
    }
});

</script>
</body>
</html>"""

    html_content = (
        html_template
        .replace("__TREE_JSON__", tree_json)
        .replace("__STATS_JSON__", stats_json)
        .replace("__LINEAR_JSON__", linear_json)
        .replace("__COLORS_JSON__", colors_json)
        .replace("__REGISTRY_JSON__", registry_json)
        .replace("__DESIGN_CARDS_JSON__", design_cards_json)
        .replace("__SHOW_DESIGN_CARDS__", show_design_cards_json)
    )

    Path(output_html).write_text(html_content, encoding="utf-8")
    print(f"Saved designer visualization to {output_html}")
    print(
        "Nodes: {total} | Unique Seqs: {unique} | Roots: {roots}".format(
            total=stats["total_nodes"],
            unique=stats["unique_sequences"],
            roots=stats["root_count"],
        )
    )
    return output_html
