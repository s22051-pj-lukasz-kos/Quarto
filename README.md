# Narzędzia sztucznej inteligencji (NAI), gr. 74c

Autors: Emilian Murawski i Łukasz Kos

## Laboratory 1 - Quarto

Deadline: 12.,10.2023

It's an terminal implementation of Quarto board game, invented by swiss mathematician, Blaise Müller. Rules are described [here](https://www.ultraboardgames.com/quarto/game-rules.php).
We used easyAI framework for creating AI opponent.  
Our implementation of the game differs in that player one does not choose a pawn for player two.

## Laboratory 2 - Irrigation system (fuzzy logic)

Deadline: 26.10.2023

It's an intelligent irrigatin system based on fuzzy logic. For implementation we used scikit-fuzzy framework and paper "An intelligent irrigation system based on fuzzy logic
control: A case study for Moroccan oriental climate
region" by Benzaouia Mohammed *et al*.

The first version of the solution was accidentally placed in [another repository](https://github.com/s22051-pj-lukasz-kos/fuzzy-logic-irrigation).

## Laboratory 3 - Movie Recommender (machine learning)

Deadline: 16.11.2023

Movie Recommender is a Python program designed to offer personalized movie suggestions based on user preferences and collaborative filtering. It clusters movies using K-means by genres, builds a user-movie matrix with ratings, and utilizes K-nearest neighbors algorithm to find similar users for recommendation generation.

## Laboratory 4 - Classification (Decision Tree and Support Vector Machine)

Deadline: 30.11.2023

Classification of two problems using Decision Tree and Support Vector Machine:

- classification of wheat seeds on class 1, 2 and 3 that take into account perimeter of seed, compactness, length of kernel, width of kernel, asymmetry coefficient and length of kernel groove.
- classification of the occurrence of heart disease that takes into account age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise induced angina and ST depression induced by exercise relative to rest.

## Laboratory 5 - Neural Networks (TensorFlow)

Deadline: 15.12.2023

Couple of Neural Network models using TensorFlow.  
We recommend to use GPU computation for running those scripts. But for that you have to install specific stack. Those scripts were run with:

- [TensorFlow 2.13](https://www.tensorflow.org/install/pip)
  - `pip install tensorflow==2.13.0`
- [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- [cuDNN 8.6](https://developer.nvidia.com/rdp/cudnn-archive)  

Otherwise, classification of one problem with CPU only computations could take several dozen minutes.

## Laboratory 6 - Computer Vision

Deadline: 20.01.2024

- npm install (you have to install Node.js)
- npm run build
- npm run dev
