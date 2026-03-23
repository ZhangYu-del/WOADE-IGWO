WOADE-IGWO: An Improved Grey Wolf Optimization Algorithm Combining Differential Evolution and Whale Optimization for Influence Maximization in Social Networks

This repository provides the official implementation of WOADE-IGWO, a discrete hybrid metaheuristic framework for the Influence Maximization (IM) problem in social networks. The proposed method integrates Grey Wolf Optimizer (GWO), Differential Evolution (DE), and Whale Optimization Algorithm (WOA) to achieve a balanced trade-off between global exploration and local exploitation. A Degree Centrality and Probabilistic Adaptive initialization strategy (DCPA) is employed to enhance the quality and diversity of the initial population. During the iterative process, a multi-strategy collaborative update mechanism dynamically combines DE-based search, whale-inspired search, and weighted grey wolf guidance. In addition, a two-hop neighborhood-based local search is introduced to refine candidate seed sets, thereby improving solution quality and convergence efficiency. This repository includes the complete algorithm implementation and supports full experimental reproducibility.

Experiments are conducted on five real-world social networks: Blog represents communication relationships among blog users; Wiki-Vote captures voting interactions in Wikipedia administrator elections; CA-HepTh denotes a collaboration network in high energy physics theory; NetHEPT corresponds to an academic co-authorship network from arXiv; Soc-Epinions models trust relationships among users. All datasets are publicly available from SNAP and related repositories.

The core implementation is provided in WOADE-IGWO.py, which encapsulates the complete optimization pipeline, including DCPA-based initialization, hybrid position update strategies, fitness evaluation based on Expected Diffusion Value (EDV), and the two-hop neighborhood local search module. The framework can be directly applied to influence maximization tasks on large-scale networks.

The algorithm follows a multi-strategy cooperative optimization paradigm. The initialization stage enhances population diversity and solution quality via DCPA. The iterative phase adaptively switches between global exploration (DE-based and stochastic search) and local exploitation (whale encircling and grey wolf-guided search). A final local refinement stage further improves the seed set through two-hop neighborhood exploration, enabling the algorithm to escape local optima and achieve higher solution precision.

The code is developed in Python 3.8+ and depends on standard libraries including networkx and numpy. The influenceModel package is optional for IC and LT diffusion simulations. The implementation runs on standard CPU architectures without requiring GPU acceleration.

To use the code, install the dependencies via pip install networkx numpy, prepare the input graph in edge-list format, configure parameters such as seed set size, population size, and maximum iterations in the main script, and execute python WOADE-IGWO.py. A complete usage example is provided in the main entry of the script.

If you use this repository in your research, please cite the following manuscript:

@article{zhang2026woadeigwo,
  title={WOADE-IGWO: An Improved Grey Wolf Optimization Algorithm Combining Differential Evolution and Whale Optimization for Influence Maximization in Social Networks},
  author={Zhang, Yu and Li, Huan and Mo, Xinyue and Na, Xiaoyu and Zeng, Xianhong},
  year={2026},
  note={Manuscript under review}
}

This project is released under the MIT License. Contributions via issues and pull requests are welcome.
