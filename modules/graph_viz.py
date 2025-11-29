from streamlit_agraph import Node, Edge, Config


def build_graph(data):
    """
    Constructs a knowledge graph from clinical trial data.

    Args:
        data (list): List of study metadata dictionaries.

    Returns:
        tuple: (nodes, edges, config) for streamlit-agraph.
    """
    nodes = []
    edges = []

    # Sets to track unique entities
    study_ids = set()
    sponsors = set()
    conditions = set()

    for study in data:
        nct_id = study.get("nct_id", "Unknown")
        title = study.get("title", "Unknown")
        sponsor = study.get("org", "Unknown")
        condition_str = study.get("condition", "")

        # 1. Study Node
        if nct_id not in study_ids:
            nodes.append(
                Node(
                    id=nct_id,
                    label=nct_id,
                    size=20,
                    color="#4B8BBE",  # Blue
                    title=title,
                    shape="dot",
                )
            )
            study_ids.add(nct_id)

        # 2. Sponsor Node & Edge
        if sponsor and sponsor != "Unknown":
            if sponsor not in sponsors:
                nodes.append(
                    Node(
                        id=sponsor,
                        label=sponsor,
                        size=15,
                        color="#FF6B6B",  # Red
                        shape="triangle",
                    )
                )
                sponsors.add(sponsor)

            # Edge: Study -> Sponsor
            edges.append(
                Edge(
                    source=nct_id, target=sponsor, label="sponsored_by", color="#CCCCCC"
                )
            )

        # 3. Condition Nodes & Edges
        if condition_str:
            conds = [c.strip() for c in condition_str.split(",") if c.strip()]
            for cond in conds:
                if cond not in conditions:
                    nodes.append(
                        Node(
                            id=cond,
                            label=cond,
                            size=15,
                            color="#6BCB77",  # Green
                            shape="diamond",
                        )
                    )
                    conditions.add(cond)

                # Edge: Study -> Condition
                edges.append(
                    Edge(source=nct_id, target=cond, label="studies", color="#CCCCCC")
                )

    # Configuration
    config = Config(
        width=800,
        height=600,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
    )

    return nodes, edges, config
