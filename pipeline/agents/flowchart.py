from typing import List, Tuple, Optional, Set, Dict
from collections import deque
import re
from smolagents import tool

class Edge:
    """
    Represents a directed edge in the flowchart with an optional Yes/No condition.

    Attributes:
        to_node (str): The identifier of the destination node
        condition (Optional[bool]): The edge condition - True for Yes, False for No, None for unconditional
    """
    def __init__(self, to_node: str, condition: Optional[bool] = None):
        self.to_node = to_node
        self.condition = condition

    def describe(self) -> str:
        """
        Returns a human-readable description of the edge condition.

        Returns:
            str: "Yes" for True condition, "No" for False condition, or empty string for unconditional
        """
        if self.condition is True:
            return "Yes"
        elif self.condition is False:
            return "No"
        return ""

class Node:
    """
    Represents a node in the flowchart with an ID, statement, and outgoing edges.

    Attributes:
        id (str): Unique identifier for the node
        statement (str): The content/statement associated with the node
        edges (List[Edge]): List of outgoing edges from this node
    """
    def __init__(self, id: str, statement: str):
        self.id = id
        self.statement = statement
        self.edges: List[Edge] = []

    def add_edge(self, to_node: str, condition: Optional[bool] = None):
        """
        Adds an outgoing edge from this node.

        Args:
            to_node (str): Identifier of the destination node
            condition (Optional[bool]): Edge condition - True for Yes, False for No, None for unconditional
        """
        self.edges.append(Edge(to_node, condition))

class FlowChart:
    """
    A flowchart implementation supporting various graph operations and path finding algorithms.

    The flowchart is represented as a directed graph where nodes can have statements and edges
    can have Yes/No conditions. This implementation supports various graph operations including
    traversal algorithms, path finding, and node relationship analysis.
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}

    def add_node(self, id: str, statement: str) -> Node:
        """
        Adds a new node to the flowchart.

        Args:
            id (str): Unique identifier for the node
            statement (str): The content/statement associated with the node

        Returns:
            Node: The newly created node object

        Raises:
            ValueError: If a node with the given id already exists
        """
        if id in self.nodes:
            raise ValueError(f"Node with id {id} already exists")
        node = Node(id, statement)
        self.nodes[id] = node
        return node

    def add_edge(self, from_id: str, to_id: str, condition: Optional[bool] = None):
        """
        Adds a directed edge between two nodes with an optional condition.

        Args:
            from_id (str): Identifier of the source node
            to_id (str): Identifier of the destination node
            condition (Optional[bool]): Edge condition - True for Yes, False for No, None for unconditional

        Raises:
            ValueError: If either the source or destination node doesn't exist
        """
        if from_id not in self.nodes or to_id not in self.nodes:
            raise ValueError(f"Node {from_id} or {to_id} not found")
        self.nodes[from_id].add_edge(to_id, condition)

    def get_statement(self, node_id: str) -> str:
        """
        Returns the statement associated with a node.

        Args:
            node_id (str): Identifier of the node

        Returns:
            str: The statement associated with the node

        Raises:
            KeyError: If the node doesn't exist
        """
        return self.nodes[node_id].statement

    def to_mermaid(self) -> str:
        """
        Converts the flowchart to Mermaid diagram syntax.

        Generates Mermaid-compatible diagram code that includes:
        - Flowchart direction (TD - top down)
        - Node definitions with appropriate shapes based on their role
        - Edge definitions with conditions as labels

        Returns:
            str: A string containing the Mermaid diagram code representing the flowchart

        Example:
            >>> fc = FlowChart()
            >>> fc.add_node("A", "Start")
            >>> fc.add_node("B", "Process")
            >>> fc.add_edge("A", "B", True)
            >>> print(fc.to_mermaid())
            flowchart TD
                A(["Start"])
                B["Process"]
                A -->|Yes| B
        """
        mermaid_lines = ["flowchart TD"]

        # Helper function to determine if a node is a decision node
        def is_decision_node(node_id: str) -> bool:
            node = self.nodes[node_id]
            has_conditional_edges = any(edge.condition is not None for edge in node.edges)
            return has_conditional_edges

        # Helper function to determine if a node is start/end
        def is_terminal_node(node_id: str) -> bool:
            node = self.nodes[node_id]
            in_degree = sum(1 for n in self.nodes.values()
                          for e in n.edges if e.to_node == node_id)
            return in_degree == 0 or len(node.edges) == 0

        # Add nodes with appropriate shapes
        for node_id, node in self.nodes.items():
            escaped_statement = node.statement.replace('"', '\\"')
            if is_decision_node(node_id):
                # Decision nodes use rhombus shape
                node_def = f'    {node_id}{{"{ escaped_statement}"}}'
            elif is_terminal_node(node_id):
                # Start/End nodes use stadium shape
                node_def = f'    {node_id}(["{ escaped_statement}"])'
            else:
                # Regular nodes use rectangle shape
                node_def = f'    {node_id}["{ escaped_statement}"]'
            mermaid_lines.append(node_def)

        # Add edges with conditions
        for from_id, node in self.nodes.items():
            for edge in node.edges:
                edge_line = f"    {from_id} -->"
                if edge.condition is not None:
                    condition_str = "Yes" if edge.condition else "No"
                    edge_line += f"|{condition_str}|"
                edge_line += f" {edge.to_node}"
                mermaid_lines.append(edge_line)

        return "\n".join(mermaid_lines)
    
    @classmethod
    def from_mermaid(cls, mermaid_code: str) -> 'FlowChart':
        """
        Creates a FlowChart instance from Mermaid diagram code.

        Supports various Mermaid node shapes and edge types:
        - Regular nodes: []
        - Round nodes: ()
        - Stadium nodes: ([])
        - Asymmetric nodes: [/]
        - Rhombus nodes: {}
        - Edge labels: -->|Label|

        Args:
            mermaid_code (str): String containing the Mermaid diagram code

        Returns:
            FlowChart: A new FlowChart instance constructed from the Mermaid code

        Raises:
            ValueError: If the Mermaid code contains invalid node or edge definitions
        """
        flowchart = cls()

        # Clean up the code
        lines = [line.strip() for line in mermaid_code.split('\n') if line.strip()]

        # Enhanced patterns for different node shapes
        node_patterns = [
            # Stadium shape: A(["text"])
            r'(\w+)\(\["([^"]+)"\]\)',
            # Asymmetric shape: A[/"text"/]
            r'(\w+)\[/"([^"]+)"/\]',
            # Rhombus shape: A{"text"}
            r'(\w+)\{"([^"]+)"\}',
            # Regular shape: A["text"]
            r'(\w+)\["([^"]+)"\]',
            # Round shape: A("text")
            r'(\w+)\("([^"]+)"\)',
        ]

        # Enhanced edge pattern
        # edge_pattern = r'(\w+)\s*-->\s*(?:\|([^|]+)\|)?\s*(\w+)'
        edge_pattern = r'(\w+).*?-->\s*(?:\|([^|]+)\|)?\s*(\w+)'




        # First pass: Create nodes
        for line in lines:
          for pattern in node_patterns:
              for node_match in re.finditer(pattern, line):  # Use finditer to match multiple nodes
                  node_id, statement = node_match.groups()
                  if node_id not in flowchart.nodes:
                      flowchart.add_node(node_id, statement)


        # Second pass: Create edges
        for line in lines:
            edge_match = re.search(edge_pattern, line)
            if edge_match:
                from_id, label, to_id = edge_match.groups()

                # Convert Yes/No labels to conditions
                condition = None
                if label:
                    label = label.strip()
                    label = label.replace('"', '')
                    if label.lower() == "yes":
                        condition = True
                    elif label.lower() == "no":
                        condition = False

                flowchart.add_edge(from_id, to_id, condition)

        return flowchart




# Define tools: they assume an instantiated 'flowchart'
def _is_valid_edge(from_node_id: str, edge: Edge, conditions: Dict[str, bool] = None) -> bool:
    """Checks if an edge satisfies the given conditions."""
    if conditions is None:
        return True
    if from_node_id in conditions:
        return edge.condition == conditions[from_node_id]
    return True

@tool
def get_statement(node_id: str) -> str:
    """
    Returns the statement associated with a node.

    Args:
        node_id: Identifier of the node

    Returns:
        str: The statement associated with the node

    Raises:
        KeyError: If the node doesn't exist
    """
    return flowchart.nodes[node_id].statement

@tool
def get_neighbours(node_id: str, include_statements: bool = False) -> str:
    """
    Returns a description of all neighboring nodes connected by outgoing edges.

    Args:
        node_id: Identifier of the node
        include_statements: (bool, optional) Whether to include node statements in the output.
            Defaults to False. Pass True if the node statements will help you make better decisions or solve the problem.

    Returns:
        str: A human-readable description of neighboring nodes, including edge conditions
            and optionally node statements

    Raises:
        KeyError: If the node doesn't exist
    """
    node = flowchart.nodes[node_id]
    neighbours = []
    for edge in node.edges:
        dest_node = flowchart.nodes[edge.to_node]
        if include_statements:
            desc = f"{dest_node.id} ({dest_node.statement})"
        else:
            desc = dest_node.id
        if edge.condition is not None:
            desc += f" [{edge.describe()}]"
        neighbours.append(desc)

    if not neighbours:
        return f"Node {node_id} has no outgoing connections"
    return f"Node {node_id} connects to: {', '.join(neighbours)}"

@tool
def in_degree(node_id: str) -> str:
    """
    Returns the number of incoming edges to a node.

    Args:
        node_id:  (str) Identifier of the node

    Returns:
        str: A description of how many incoming connections the node has

    Raises:
        KeyError: If the node doesn't exist
    """
    count = sum(1 for node in flowchart.nodes.values()
                for edge in node.edges if edge.to_node == node_id)
    return f"Node {node_id} has {count} incoming connections"

@tool
def get_ancestors(node_id: str, levels: Optional[int] = None,
                    include_statements: bool = False) -> str:
    """
    Returns all nodes that have paths leading to the given node.

    Args:
        node_id: Identifier of the target node
        levels:  (Optional[int]) Maximum number of levels to traverse upward.
            If None, traverses all levels. Eg.levels = 1, if you want to view immediate ancestor of the node.
        include_statements: (bool, optional) Whether to include node statements in the output.
            Defaults to False. Pass True if the node statements will help you make better decisions or solve the problem.

    Returns:
        str: A description of all ancestor nodes, optionally limited by levels and
            including node statements

    Raises:
        KeyError: If the node doesn't exist
    """
    ancestors = set()
    current_level = {node_id}
    level_count = 0

    while current_level and (levels is None or level_count < levels):
        next_level = set()
        for current in flowchart.nodes.values():
            for edge in current.edges:
                if edge.to_node in current_level and current.id not in ancestors:
                    next_level.add(current.id)
        ancestors.update(next_level)
        current_level = next_level
        level_count += 1

    if not ancestors:
        return f"Node {node_id} has no ancestors"

    ancestor_desc = []
    for anc_id in ancestors:
        if include_statements:
            ancestor_desc.append(f"{anc_id} ({flowchart.nodes[anc_id].statement})")
        else:
            ancestor_desc.append(anc_id)

    level_desc = f" within {levels} levels" if levels is not None else ""
    return f"Ancestors of node {node_id}{level_desc}: {', '.join(ancestor_desc)}"

@tool    
def path_between(start_id: str, end_id: str,
                conditions: Optional[Dict[str, bool]] = None,
                include_statements: bool = False) -> str:
    """
    Finds a path between two nodes, considering edge conditions.

    Args:
        start_id: Identifier of the starting node
        end_id: Identifier of the destination node
        conditions: (Optional[Dict[str, bool]]) Dictionary mapping node IDs to required
            edge conditions (True for Yes, False for No). Example: 'C' is a decision node, where the response to 'C' is 'No', so you can pass {'C': False}. Ignore this parameter if you are not concerned with the values of the edges. 
        include_statements:  (bool, optional) Whether to include node statements in the output.
            Defaults to False. Pass True if the node statements will help you make better decisions or solve the problem. 

    Returns:
        str: A description of the path found, including edge conditions and optionally
            node statements. If no path exists, returns a message indicating this.

    Raises:
        KeyError: If either the start or end node doesn't exist
    """
    visited = set()
    path = []

    def _dfs_path(current_id: str) -> bool:
        if current_id == end_id:
            path.append(current_id)
            return True

        if current_id not in visited:
            visited.add(current_id)
            node = flowchart.nodes[current_id]

            for edge in node.edges:
                if _is_valid_edge(node.id, edge, conditions):
                    if _dfs_path(edge.to_node):
                        path.append(current_id)
                        return True
        return False

    if not _dfs_path(start_id):
        return f"No path exists between {start_id} and {end_id}"

    path = list(reversed(path))
    path_desc = []

    for i, node_id in enumerate(path):
        if include_statements:
            node_desc = f"{node_id} ({flowchart.nodes[node_id].statement})"
        else:
            node_desc = node_id
        path_desc.append(node_desc)

        if i < len(path) - 1:
            next_id = path[i + 1]
            for edge in flowchart.nodes[node_id].edges:
                if edge.to_node == next_id:
                    if edge.condition is not None:
                        path_desc.append(f"[{edge.describe()}]")

    return f"Path found: {' -> '.join(path_desc)}"

@tool
def get_neighbours(node_id: str, include_statements: bool = False) -> str:
    """Returns a description of all nodes connected to the given node by outgoing edges.

    Args:
        node_id:  (str) The identifier of the node whose neighbors we want to find.
        include_statements: (bool, optional) If True, includes the statement text for each
            neighboring node in the output. Defaults to False.

    Returns:
        str: A formatted string describing all neighboring nodes, including any edge
            conditions if present. If the node has no outgoing connections, returns a
            message indicating this.

    Example:
        >>> get_neighbours("A", include_statements=True)
        'Node A connects to: B (Check inventory) [Yes], C (Contact supplier) [No]'
    """
    node = flowchart.nodes[node_id]
    neighbours = []
    for edge in node.edges:
        dest_node = flowchart.nodes[edge.to_node]
        if include_statements:
            desc = f"{dest_node.id} ({dest_node.statement})"
        else:
            desc = dest_node.id
        if edge.condition is not None:
            desc += f" [{edge.describe()}]"
        neighbours.append(desc)
    
    if not neighbours:
        return f"Node {node_id} has no outgoing connections"
    return f"Node {node_id} connects to: {', '.join(neighbours)}"

@tool
def in_degree(node_id: str) -> str:
    """Calculates and returns the number of incoming edges to a specified node.

    Args:
        node_id: (str) The identifier of the node to analyze.

    Returns:
        str: A formatted string indicating the number of incoming connections
            to the specified node.

    Example:
        >>> in_degree("B")
        'Node B has 2 incoming connections'
    """
    count = sum(1 for node in flowchart.nodes.values() 
                for edge in node.edges if edge.to_node == node_id)
    return f"Node {node_id} has {count} incoming connections"

@tool
def out_degree(node_id: str) -> str:
    """Calculates and returns the number of outgoing edges from a specified node.

    Args:
        node_id: (str) The identifier of the node to analyze.

    Returns:
        str: A formatted string indicating the number of outgoing connections
            from the specified node.

    Example:
        >>> out_degree("A")
        'Node A has 3 outgoing connections'
    """
    count = len(flowchart.nodes[node_id].edges)
    return f"Node {node_id} has {count} outgoing connections"

@tool
def max_in_degree() -> str:
    """Identifies all nodes with the highest number of incoming edges in the flowchart.

    Returns:
        str: A formatted string identifying all nodes with the most incoming
            connections and their in-degree count.

    Example:
        >>> max_in_degree()
        'Nodes D, E have the highest number of incoming connections (5)'
    """
    max_degree = -1
    max_nodes = []
    for node_id in flowchart.nodes:
        degree = sum(1 for n in flowchart.nodes.values() 
                    for e in n.edges if e.to_node == node_id)
        if degree > max_degree:
            max_degree = degree
            max_nodes = [node_id]
        elif degree == max_degree:
            max_nodes.append(node_id)
    return f"Nodes {', '.join(max_nodes)} have the highest number of incoming connections ({max_degree})"

@tool
def max_out_degree() -> str:
    """Identifies all nodes with the highest number of outgoing edges in the flowchart.

    Returns:
        str: A formatted string identifying all nodes with the most outgoing
            connections and their out-degree count.

    Example:
        >>> max_out_degree()
        'Nodes A, B have the highest number of outgoing connections (4)'
    """
    max_degree = -1
    max_nodes = []
    for node_id, node in flowchart.nodes.items():
        degree = len(node.edges)
        if degree > max_degree:
            max_degree = degree
            max_nodes = [node_id]
        elif degree == max_degree:
            max_nodes.append(node_id)
    return f"Nodes {', '.join(max_nodes)} have the highest number of outgoing connections ({max_degree})"

@tool
def get_ancestors(node_id: str, levels: int = None, include_statements: bool = False) -> str:
    """Identifies all nodes that have paths leading to the specified node.

    Args:
        node_id: (str) The identifier of the target node.
        levels: (int, optional) Maximum number of levels to traverse upward.
            If None, traverses all possible levels. Defaults to None.
        include_statements: (bool, optional) If True, includes the statement text
            for each ancestor node in the output. Defaults to False.

    Returns:
        str: A formatted string listing all ancestor nodes of the specified node,
            optionally limited by levels and including statements.

    Example:
        >>> get_ancestors("D", levels=2, include_statements=True)
        'Ancestors of node D within 2 levels: B (Check inventory), C (Contact supplier)'
    """
    ancestors = set()
    current_level = {node_id}
    level_count = 0
    
    while current_level and (levels is None or level_count < levels):
        next_level = set()
        for current in flowchart.nodes.values():
            for edge in current.edges:
                if edge.to_node in current_level and current.id not in ancestors:
                    next_level.add(current.id)
        ancestors.update(next_level)
        current_level = next_level
        level_count += 1
    
    if not ancestors:
        return f"Node {node_id} has no ancestors"
    
    ancestor_desc = []
    for anc_id in ancestors:
        if include_statements:
            ancestor_desc.append(f"{anc_id} ({flowchart.nodes[anc_id].statement})")
        else:
            ancestor_desc.append(anc_id)
    
    level_desc = f" within {levels} levels" if levels is not None else ""
    return f"Ancestors of node {node_id}{level_desc}: {', '.join(ancestor_desc)}"

@tool
def get_descendants(node_id: str, levels: int = None, include_statements: bool = False) -> str:
    """Identifies all nodes that can be reached from the specified node.

    Args:
        node_id: (str) The identifier of the starting node.
        levels: (int, optional) Maximum number of levels to traverse downward.
            If None, traverses all possible levels. Defaults to None.
        include_statements: (bool, optional) If True, includes the statement text
            for each descendant node in the output. Defaults to False.

    Returns:
        str: A formatted string listing all descendant nodes of the specified node,
            optionally limited by levels and including statements.

    Example:
        >>> get_descendants("A", levels=1, include_statements=True)
        'Descendants of node A within 1 level: B (Check inventory), C (Contact supplier)'
    """
    descendants = set()
    current_level = {node_id}
    level_count = 0
    
    while current_level and (levels is None or level_count < levels):
        next_level = set()
        for current_id in current_level:
            for edge in flowchart.nodes[current_id].edges:
                if edge.to_node not in descendants:
                    next_level.add(edge.to_node)
        descendants.update(next_level)
        current_level = next_level
        level_count += 1
    
    if not descendants:
        return f"Node {node_id} has no descendants"
    
    desc_desc = []
    for desc_id in descendants:
        if include_statements:
            desc_desc.append(f"{desc_id} ({flowchart.nodes[desc_id].statement})")
        else:
            desc_desc.append(desc_id)
    
    level_desc = f" within {levels} levels" if levels is not None else ""
    return f"Descendants of node {node_id}{level_desc}: {', '.join(desc_desc)}"

@tool
def bfs(start_id: str = None, conditions: Dict[str, bool] = None, 
        include_statements: bool = False) -> str:
    """Performs a breadth-first search traversal of the flowchart from a starting node.

    Args:
        start_id: (str, optional) The identifier of the starting node. If None,
            uses the first node in the flowchart. Defaults to None.
        conditions: (Dict[str, bool], optional) Dictionary mapping node IDs to required
            edge conditions (True for Yes, False for No). For example, if 'C' is a
            decision node and you want to follow the 'No' path, pass {'C': False}.
            Defaults to None.
        include_statements: (bool, optional) If True, includes the statement text
            for each node in the traversal path. Defaults to False.

    Returns:
        str: A formatted string describing the BFS traversal path through the
            flowchart, respecting any specified edge conditions.

    Example:
        >>> bfs("A", conditions={"C": False}, include_statements=True)
        'BFS traversal path: A (Start) -> B (Check inventory) -> D (Update system)'
    """
    if start_id is None:
        start_id = next(iter(flowchart.nodes))
    
    visited = []
    queue = deque([start_id])
    visited_set = set()
    
    while queue:
        node_id = queue.popleft()
        if node_id not in visited_set:
            visited_set.add(node_id)
            visited.append(node_id)
            
            node = flowchart.nodes[node_id]
            for edge in node.edges:
                if _is_valid_edge(node.id, edge, conditions):
                    queue.append(edge.to_node)
    
    path_desc = []
    for node_id in visited:
        if include_statements:
            path_desc.append(f"{node_id} ({flowchart.nodes[node_id].statement})")
        else:
            path_desc.append(node_id)
    
    return f"BFS traversal path: {' -> '.join(path_desc)}"

@tool
def shortest_path(start_id: str, end_id: str, 
                     conditions: Optional[Dict[str, bool]] = None,
                     include_statements: bool = False) -> str:
        """
        Finds the shortest path between two nodes using BFS, considering edge conditions.
        
        Args:
            start_id: Identifier of the starting node
            end_id: Identifier of the destination node
            conditions: (Optional[Dict[str, bool]]) Dictionary mapping node IDs to required
                edge conditions (True for Yes, False for No)
            include_statements: (bool, optional) Whether to include node statements in the output.
                Defaults to False.
        
        Returns:
            str: A description of the shortest path found, including edge conditions and 
                optionally node statements. If no path exists, returns a message indicating this.
        
        Raises:
            KeyError: If either the start or end node doesn't exist
        """
        if start_id not in flowchart.nodes or end_id not in flowchart.nodes:
            raise KeyError(f"Start node {start_id} or end node {end_id} not found")
            
        # Initialize BFS data structures
        queue = deque([(start_id, [])])  # (node_id, path_so_far)
        visited = {start_id}
        
        while queue:
            current_id, path = queue.popleft()
            path = path + [current_id]
            
            if current_id == end_id:
                # Path found, format the output
                path_desc = []
                for i, node_id in enumerate(path):
                    if include_statements:
                        node_desc = f"{node_id} ({flowchart.nodes[node_id].statement})"
                    else:
                        node_desc = node_id
                    path_desc.append(node_desc)
                    
                    # Add edge conditions if there's a next node
                    if i < len(path) - 1:
                        next_id = path[i + 1]
                        for edge in flowchart.nodes[node_id].edges:
                            if edge.to_node == next_id:
                                if edge.condition is not None:
                                    path_desc.append(f"[{edge.describe()}]")
                
                steps = len(path) - 1
                return f"Shortest path found ({steps} steps): {' -> '.join(path_desc)}"
            
            # Explore neighbors
            node = flowchart.nodes[current_id]
            for edge in node.edges:
                next_id = edge.to_node
                if next_id not in visited and _is_valid_edge(node.id, edge, conditions):
                    visited.add(next_id)
                    queue.append((next_id, path))
        
        return f"No path exists between {start_id} and {end_id}"

@tool    
def dfs(start_id: Optional[str] = None, 
        conditions: Optional[Dict[str, bool]] = None,
        include_statements: bool = False) -> str:
    """
    Performs depth-first search traversal of the flowchart.
    
    Args:
        start_id: (Optional[str]) Identifier of the starting node. If None, starts
            from the first node in the flowchart.
        conditions: (Optional[Dict[str, bool]]) Dictionary mapping node IDs to required
            edge conditions (True for Yes, False for No)
        include_statements: (bool, optional) Whether to include node statements in the output.
            Defaults to False.
    
    Returns:
        str: A description of the DFS traversal path, optionally including node statements
            and edge conditions.
    
    Raises:
        KeyError: If the start node doesn't exist
        ValueError: If the flowchart is empty
    """
    if not flowchart.nodes:
        raise ValueError("Flowchart is empty")
        
    if start_id is None:
        start_id = next(iter(flowchart.nodes))
    elif start_id not in flowchart.nodes:
        raise KeyError(f"Start node {start_id} not found")
    
    visited = set()
    path = []
    edge_conditions = []  # Store edge conditions between nodes
    
    def _dfs_helper(node_id: str):
        visited.add(node_id)
        path.append(node_id)
        
        node = flowchart.nodes[node_id]
        for edge in node.edges:
            if _is_valid_edge(edge, conditions):
                next_id = edge.to_node
                if next_id not in visited:
                    # Store edge condition if it exists
                    if edge.condition is not None:
                        edge_conditions.append((len(path) - 1, edge.describe()))
                    _dfs_helper(next_id)
    
    _dfs_helper(start_id)
    
    # Format the output
    path_desc = []
    for i, node_id in enumerate(path):
        if include_statements:
            node_desc = f"{node_id} ({flowchart.nodes[node_id].statement})"
        else:
            node_desc = node_id
        path_desc.append(node_desc)
        
        # Add edge condition if it exists for this position
        for pos, condition in edge_conditions:
            if pos == i:
                path_desc.append(f"[{condition}]")
    
    return f"DFS traversal path: {' -> '.join(path_desc)}"



