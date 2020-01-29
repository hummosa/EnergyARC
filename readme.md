# Energy-based model for reasoning and transfer Reinforcement learning


## Introduction:
Energy Based Models are a choice to model complex data and environments using one high-dimensional distribution, referred to as an energy surface. Realistic or desired data points are mapped to the valleys of the surface while unlikely data points are mapped to the peaks. Algorithms can now be devised to interact with this surface and influence its landscape. It easily admits other formulations to interface with the energy surface, and in this work we embed an RL agent as well as a dynamical sampler and allow them both to explore and influence its topology.  The goal of this work is to answer the question: can 'reasoning' be implemented by carving the slopes of the energy surface? Can we map abstract conceptual operations to the valleys and hills of the energy landscape?

Towards answering the question, we tackle the challenging Abstract Reasoning Corpus (ARC) dataset [1], released recently, to serve as a benchmark for reasoning agents. ARC comprises of 800 tasks on a 2D grid,  split into non-overlapping training and testing sets. The tasks are challenging and draw upon a wide range of priors about objects, dynamics, intentions, and perceptual patterns. For each task the agent has to experiment, sometimes extensively, and is required to examine a few demonstration input-output pairs and then infer the correct output to a test input.

The energy surface sampler takes several (reasoning) steps across the surface before producing an initial guess of how to perceive the task and the workplan to solve it. The RL agent takes this initial guess and explore its neighborhood. As the RL agent painstakingly finds solutions to more tasks, the sampler begins to provide better initial guesses for subsequent ones.

The tasks are challenging and designed to require symbolic search similar to automated theorem proving systems, but here we devise a learning-based system which relies on these novel components:
    - The demonstration correct outputs are never shown to the system, but rather converted into a reward signal for the RL agent as it tries to solve them.
    - The RL agent has the ability to explore not only actions, but also different ways of perceiving the input grid.
    - Following [2], we similarly use a task embedding that biases the model to form useful abstractions of tasks and objects in them.

## Requirements:

- [ARC](https://github.com/fchollet/ARC/tree/1f68da7cf7c5b1849cef67f0e2d74680b42306a8)
- pytorch 1.4
- wandb (to visualize model activations and weight distributions) 
- matplotlib
- graphviz (for debugging)
- pillow <= 6.2.0


## Training:
To train the EBM and the sampler generatively, run trainarc.py. The RL agent, coming up soon, is however necessary to be able to solve any tasks, as the labels (correct output) are not given to the model at any point. It merely learns typical patterns of inputs and outputs. 

## Results:
Currently the energy model, and the sampler have been implemented and are able to learn a generative model of the task input output pairs. Now implementing the RL agent. 

## Next steps:
Several previous works have integrated RL into an EBM formulation [3,4,5,6]. More recently, [7] showed that such integration does improve the ability of the model to transfer learning to novel tasks. 

## References:
[1] François Chollet. On the measure of intelligence, 2019.

[2] Igor Mordatch. Concept learning with energy-based models, 2018.

[3] Brian Sallans, Geoffrey E. Hinton, and Sridhar Mahadevan. Reinforcement learning with factored states and actions.Journal of Machine Learning Research, 5:1063–1088, 2004.[4]Makoto Otsuka, Junichiro Yoshimoto, and Kenji Doya. Free-energy-based reinforcement learning in a partiallyobservable environment.

[5] Nicolas Heess, David Silver, Yee Whye Teh, Peter Deisenroth, Csaba Szepesvári, and Jan Peters.  Actor-criticreinforcement learning with energy-based policies.

[6] Stefan Elfwing, Makoto Otsuka, Eiji Uchibe, and Kenji Doya.  Free-energy based reinforcement learning forvision-based navigation with high-dimensional sensory inputs.  In Kok Wai Wong, B. Sumudu U. Mendis, andAbdesselam Bouzerdoum, editors,ICONIP (1), volume 6443 ofLecture Notes in Computer Science, pages 215–222.Springer, 2010.

[7] Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-basedpolicies.CoRR, abs/1702.08165, 2017.
