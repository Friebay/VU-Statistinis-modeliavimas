import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple
import random

def create_transition_matrix(edges: List[Tuple[int, int]], num_vertices: int) -> np.ndarray:
    """
    Create a transition probability matrix from a list of edges.
    
    Args:
        edges: List of tuples representing directed edges (from_vertex, to_vertex)
        num_vertices: Number of vertices in the graph
    
    Returns:
        Transition probability matrix
    """
    # Initialize transition matrix with zeros
    transition_matrix = np.zeros((num_vertices, num_vertices))
    
    # Count outgoing edges for each vertex
    outgoing_counts = [0] * num_vertices
    for from_vertex, _ in edges:
        outgoing_counts[from_vertex - 1] += 1
    
    # Fill in transition probabilities
    for from_vertex, to_vertex in edges:
        # Adjust for 0-indexed matrix
        from_idx = from_vertex - 1
        to_idx = to_vertex - 1
        
        # Probability is 1 / (number of outgoing edges)
        if outgoing_counts[from_idx] > 0:
            transition_matrix[from_idx, to_idx] = 1 / outgoing_counts[from_idx]
    
    return transition_matrix

def generate_markov_chain(transition_matrix: np.ndarray, starting_state: int, num_steps: int) -> List[int]:
    """
    Generate a Markov chain using the given transition matrix.
    
    Args:
        transition_matrix: The transition probability matrix
        starting_state: Initial state (0-indexed)
        num_steps: Number of steps to simulate
        
    Returns:
        List of states in the Markov chain
    """
    num_states = transition_matrix.shape[0]
    chain = [starting_state]
    
    current_state = starting_state
    for _ in range(num_steps):
        # Get probability distribution for current state
        probabilities = transition_matrix[current_state]
        
        # If there are no outgoing transitions (absorbing state), stay in the current state
        if np.sum(probabilities) < 1e-10:  # Check if probabilities sum to approximately zero
            next_state = current_state
        else:
            # Choose next state based on probabilities
            # Ensure probabilities sum to 1 to avoid numerical issues
            probabilities = probabilities / np.sum(probabilities)
            next_state = np.random.choice(num_states, p=probabilities)
        
        chain.append(next_state)
        current_state = next_state
    
    # Convert to 1-indexed for output
    return [state + 1 for state in chain]

def visualize_graph(edges: List[Tuple[int, int]], num_vertices: int, transition_matrix: np.ndarray) -> None:
    """
    Visualize the graph using NetworkX.
    
    Args:
        edges: List of tuples representing directed edges
        num_vertices: Number of vertices in the graph
        transition_matrix: The transition probability matrix
    """
    G = nx.DiGraph()
    
    # Add all vertices
    for i in range(1, num_vertices + 1):
        G.add_node(i, label=f"v_{i}")
    
    # Add edges
    G.add_edges_from(edges)
    
    # Draw the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=123)  # Position nodes using spring layout with a fixed seed
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels={i: f"v_{i}" for i in range(1, num_vertices+1)}, font_size=12)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowsize=20)
    
    # Calculate edge labels based on transition probabilities
    edge_labels = {}
    for from_vertex, to_vertex in edges:
        prob = transition_matrix[from_vertex-1, to_vertex-1]
        edge_labels[(from_vertex, to_vertex)] = f"{prob:.2f}"
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title("Markov Chain Graph Visualization")
    plt.axis('off')
    plt.savefig("markov_chain_graph.png")
    plt.show()

def main():
    """
    Main function to demonstrate the Markov chain generation and analysis.
    """
    print("Markov Chain Analysis for the Given Graph")
    print("========================================")
    
    # Define the graph from the problem statement
    vertices = [1, 2, 3, 4, 5]  # v₁, v₂, v₃, v₄, v₅
    edges = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (4, 5)]
    
    print("Graph Structure:")
    print(f"Vertices: S = {{v₁, v₂, v₃, v₄, v₅}}")
    print(f"Edges: E = {{〈v₁, v₂〉, 〈v₁, v₃〉, 〈v₁, v₄〉, 〈v₁, v₅〉, 〈v₂, v₃〉, 〈v₄, v₅〉}}")
    print("\nNote: Transitions are made with equal probability to all neighbors.")
    
    # Create transition matrix
    transition_matrix = create_transition_matrix(edges, len(vertices))
    
    print("\nStep 1: Computing the Transition Probability Matrix")
    print("-------------------------------------------------")
    print("Rows represent source vertices, columns represent destination vertices.")
    print("Transition Probability Matrix:")
    print("     | v₁   | v₂   | v₃   | v₄   | v₅   |")
    print("-----|------|------|------|------|------|")
    for i, row in enumerate(transition_matrix):
        row_str = f" v₁{i+1} | " + " | ".join([f"{prob:.2f}" for prob in row]) + " |"
        print(row_str)
      # Generate multiple Markov chains
    starting_state = 0  # Start from v₁ (0-indexed)
    num_steps = 5
    num_samples = 4
    print(f"\nStep 2: Generating {num_samples} Sample Markov Chains (each with {num_steps} steps)")
    print("-------------------------------------------------")
    
    # Set seed for reproducibility but get different chains
    np.random.seed(42)
    
    for i in range(num_samples):
        chain = generate_markov_chain(transition_matrix, starting_state, num_steps)
        
        # Display the chain
        print(f"\nMarkov Chain {i+1} starting from v₁:")
        chain_str = " → ".join([f"v_{state}" for state in chain])
        print(chain_str)
    
    # Visualize the graph
    print("\nStep 3: Visualizing the Markov Chain Graph")
    print("-------------------------------------------------")
    print("(A graph visualization will appear - close it to continue)")
    visualize_graph(edges, len(vertices), transition_matrix)
    
    # Simulate many chains to check state distribution
    print("\nStep 4: Analyzing Long-Run Behavior")
    print("-------------------------------------------------")
    print("Simulating 1,000 chains of length 5 to find the stationary distribution...")
    
    np.random.seed(None)  # Reset seed for random simulations
    num_simulations = 1000
    chain_length = 5
    final_states = []
    
    for _ in range(num_simulations):
        chain = generate_markov_chain(transition_matrix, starting_state, chain_length)
        final_states.append(chain[-1])
    
    # Count occurrences of each state
    state_counts = {}
    for state in range(1, len(vertices) + 1):
        count = final_states.count(state)
        state_counts[state] = count / num_simulations
    
    print("\nLong-run frequency distribution of states:")
    for state, prob in state_counts.items():
        print(f"v_{state}: {prob:.4f}")
    
    # Identify absorbing states
    absorbing_states = []
    for i in range(len(vertices)):
        if np.sum(transition_matrix[i]) < 1e-10:
            absorbing_states.append(i + 1)
    
    print(f"\nAbsorbing states identified: {', '.join([f'v_{s}' for s in absorbing_states])}")
    print("The chain will eventually get trapped in one of these states.")

if __name__ == "__main__":
    main()