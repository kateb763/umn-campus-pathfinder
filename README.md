# Pathfinding Visualizer

A web-based **pathfinding visualizer** showcasing multiple graph search algorithms, built with **Python**, **Flask**, and **JavaScript**. Explore how different algorithms traverse a graph to find the shortest or most efficient path across the University of Minnesota campus. 

---

## 🚀 Features

- **Multiple Algorithms Implemented**
  - Breadth-First Search  
  - Depth-First Search  
  - Dijkstra
  - A* Search
  - Bellman-Ford
  - Minimum Spanning Tree  
  - Random Path / Fly / Point-to-Point for comparison
- **Interactive Visualization**
  - Input start and destination points
  - Visualizes paths on a graph
  - Supports multiple algorithm comparisons
- **Modular Python Backend**
  - Algorithms implemented in `search_algorithms.py`
  - Flask API for JSON POST requests
- **Custom Graph Support**
  - Load your own graph data via `graph.json`

---

## 📸 Screenshot

![Visualizer Screenshot](pathfinding_visualizer.png)  
*Example visualization*

---

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/kateb763/pathfinding-visualizer.git
cd pathfinding-visualizer
