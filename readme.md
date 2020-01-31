# Energy-based model for reasoning and transfer Reinforcement learning

## Introduction

Energy Based Models (EBM) are a choice to model complex data and environments using one high-dimensional distribution, referred to as an energy surface. Realistic or desired data points are mapped to the valleys of the surface while unlikely data points are mapped to the peaks. The beauty comes from how easily it admits other formulations to interface with the energy surface, including composing multiple energy models, in addition to many other mathematicall constucts. This work combines and EBM with a Langevin dynamic sampler that surfs along the energy surface gradient, and an RL agent that can explore regions of the surface by interactions with the environment. Both the sampler and the RL agent can then synergistically  explore and carve the energy landscape, make reasoning about novel tasks more effective, rapid and 'intuitive'. The goal of this work is to answer the question: can 'reasoning' be implemented by shaping the slopes of the energy surface? i.e. can we map abstract conceptual operations to the valleys and hills of the energy landscape?

Towards answering the question, we tackle the challenging Abstract Reasoning Corpus ([ARC](https://github.com/fchollet/ARC/tree/1f68da7cf7c5b1849cef67f0e2d74680b42306a8)) dataset [1], released recently, to serve as a benchmark for reasoning agents. ARC comprises of 800 tasks on a 2D grid, split into non-overlapping training and testing sets. The tasks are challenging and draw upon a wide range of priors about objects, dynamics, intentions, and perceptual patterns. For each task the agent has to experiment, sometimes extensively. The agent is examines a few demonstration input-output pairs and then infer the correct output to a test input.

The energy surface sampler takes several (reasoning) steps across the surface before producing a number of initial guesses of how to perceive the task and the workplan to solve it. The RL agent takes the initial guesses and explores their region. As the RL agent painstakingly finds solutions to more tasks, the sampler begins to provide better initial guesses for subsequent ones.

The tasks are challenging and designed to require symbolic search similar to automated theorem proving systems, but here we devise a learning-based system which relies on these novel components:

- ARC is does not contain enough data to support data-hungry deep learning methods. To compensate, the correct input-output pairs are never directly shown to the system, but rather converted into a reward signal for the RL agent as it tries to solve them.
- Search for a solution in the search space is split between the EBM that provides many good initial guesses (multiple samples), and an RL agent that further explores the suggested neighborhoods.
- The RL agent has the ability to explore not only actions, but also different ways of perceiving the input grid.
- Following [2], we learn a task embedding that biases the model to form useful abstractions of tasks and concepts in them.

## Requirements

- [ARC](https://github.com/fchollet/ARC/tree/1f68da7cf7c5b1849cef67f0e2d74680b42306a8)
- pytorch 1.4
- wandb (to visualize model activations and weight distributions)
- matplotlib
- graphviz (for debugging)
- pillow <= 6.2.0

## Training

To train the EBM and the sampler generatively, run trainarc.py. The RL agent, coming up soon, is however necessary to be able to solve any tasks, as the labels (correct output) are not given to the model at any point. It merely learns typical patterns of inputs and outputs. [Code](https://github.com/rosinality/igebm-pytorch.git) by @rosinality served as a starting point for our project. The algorithm involves taking the derivative of the energy functio with respect to inputs and coasting on the energy surface for several steps, and then taking the derivative of all those grandient steps with respect to the model parameters. The code implements methods to trace memory allocation, as well as using pytorch profiler tool to track CPU and GPU usage. Managing resources and using the simplest possible model are necessary for this approach. 

## Results

Currently the energy model, and the sampler have been implemented and are able to learn a generative model of the task input output pairs. Now implementing the RL agent.

## Next steps

Several previous works have integrated RL into an EBM formulation [3,4,5,6]. More recently, [7] showed that such integration does improve the ability of the model to transfer learning to novel tasks.

## References

[1] François Chollet. On the measure of intelligence, 2019.

[2] Igor Mordatch. Concept learning with energy-based models, 2018.

[3] Brian Sallans, Geoffrey E. Hinton, and Sridhar Mahadevan. Reinforcement learning with factored states and actions.Journal of Machine Learning Research, 5:1063–1088, 2004.[4]Makoto Otsuka, Junichiro Yoshimoto, and Kenji Doya. Free-energy-based reinforcement learning in a partiallyobservable environment.

[5] Nicolas Heess, David Silver, Yee Whye Teh, Peter Deisenroth, Csaba Szepesvári, and Jan Peters.  Actor-criticreinforcement learning with energy-based policies.

[6] Stefan Elfwing, Makoto Otsuka, Eiji Uchibe, and Kenji Doya.  Free-energy based reinforcement learning forvision-based navigation with high-dimensional sensory inputs.  In Kok Wai Wong, B. Sumudu U. Mendis, andAbdesselam Bouzerdoum, editors,ICONIP (1), volume 6443 ofLecture Notes in Computer Science, pages 215–222.Springer, 2010.

[7] Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-basedpolicies.CoRR, abs/1702.08165, 2017.
