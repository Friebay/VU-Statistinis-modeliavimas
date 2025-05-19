import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple

def create_transition_matrix(edges: List[Tuple[int, int]], num_vertices: int):
    # Pradinė matrica su nuliais
    transition_matrix = np.zeros((num_vertices, num_vertices))
    
    # Suskaičiuokite kiekvienos viršūnės išeinančias briaunas
    outgoing_counts = [0] * num_vertices
    for from_vertex, _ in edges:
        outgoing_counts[from_vertex - 1] += 1

    # Užpildykite perėjimo tikimybių matricą
    for from_vertex, to_vertex in edges:
        from_idx = from_vertex - 1
        to_idx = to_vertex - 1
        
        # Tikimybė yra 1 / (išeinančių briaunų skaičius)
        if outgoing_counts[from_idx] > 0:
            transition_matrix[from_idx, to_idx] = 1 / outgoing_counts[from_idx]
    
    return transition_matrix

def generate_markov_chain(transition_matrix: np.ndarray, starting_state: int, num_steps: int) -> List[int]:
    num_states = transition_matrix.shape[0]
    chain = [starting_state]
    
    current_state = starting_state
    for _ in range(num_steps):
        # Gauti tikimybių pasiskirstymą dabartinei būsenai
        probabilities = transition_matrix[current_state]
        
        # Jei nėra išeinančiųjų perėjimų (absorbuojanti būsena), liekame dabartinėje būsenoje
        if np.sum(probabilities) < 1e-10:  # Patikrinti, ar tikimybių suma apytiksliai lygi nuliui
            next_state = current_state
        else:
            # Pasirinkti kitą būseną pagal tikimybes
            # Užtikrinti, kad tikimybių suma būtų 1, kad išvengti skaitinių problemų
            probabilities = probabilities / np.sum(probabilities)
            next_state = np.random.choice(num_states, p=probabilities)
        
        chain.append(next_state)
        current_state = next_state
    
    # Konvertuoti į 1-indeksuotą išvedimą
    return [state + 1 for state in chain]

def visualize_graph(edges: List[Tuple[int, int]], num_vertices: int, transition_matrix: np.ndarray):
    G = nx.DiGraph()
    
    # Pridėti visas viršūnes
    for i in range(1, num_vertices + 1):
        G.add_node(i, label=f"v_{i}")
    
    # Pridėti briaunas
    G.add_edges_from(edges)
    
    # Nubrėžti grafą
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=123)
    
    # Nubrėžti viršūnes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700)
    
    # Nubrėžti viršūnių etiketes
    nx.draw_networkx_labels(G, pos, labels={i: f"v_{i}" for i in range(1, num_vertices+1)}, font_size=12)
    
    # Nubrėžti briaunas
    nx.draw_networkx_edges(G, pos, arrowsize=20)
    
    # Apskaičiuoti briaunų etiketes pagal perėjimo tikimybes
    edge_labels = {}
    for from_vertex, to_vertex in edges:
        prob = transition_matrix[from_vertex-1, to_vertex-1]
        edge_labels[(from_vertex, to_vertex)] = f"{prob:.2f}"
    
    # Nubrėžti briaunų etiketes
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title("Markovo grandinės grafo vizualizacija")
    plt.axis('off')
    plt.show()

def main():  
    # Apibrėžti grafą pagal uždavinio sąlygą
    vertices = [1, 2, 3, 4, 5]
    edges = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (4, 5)]
    
    print("Duota:")
    print(f"Viršūnės: S = {{v1, v2, v3, v4, v5}}")
    print(f"Briaunos: E = {{〈v1, v2〉, 〈v1, v3〉, 〈v1, v4〉, 〈v1, v5〉, 〈v2, v3〉, 〈v4, v5〉}}")
    print("\nPerėjimai atliekami su vienoda tikimybe į visus kaimynus.")
    
    # Sukurti perėjimo matricą
    transition_matrix = create_transition_matrix(edges, len(vertices))
    
    print("-------------------------------------------------")

    print("Perėjimo tikimybių matrica:")
    print("     | v1   | v2   | v3   | v4   | v5   |")
    print("-----|------|------|------|------|------|")
    for i, row in enumerate(transition_matrix):
        row_str = f"  v{i+1} |"
        for prob in row:
            row_str += f" {prob:.2f} |"
        print(row_str)

    # Vizualizuoti grafą
    print("\nMarkovo grandinės grafo vizualizavimas")
    print("-------------------------------------------------")
    # visualize_graph(edges, len(vertices), transition_matrix)
    

if __name__ == "__main__":
    main()