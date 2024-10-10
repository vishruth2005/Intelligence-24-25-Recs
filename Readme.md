# Intelligence-24-25-recruitment-submission - Overview

# By Vishruth-V-Srivatsa-231DS034

This repository contains the completed recruitment task, which is divided into three main tasks. Each task is further broken down into subtasks, with detailed instructions and outputs documented in individual README files within the respective task directories.

## Table of Contents

1. [Task 1 - Kaggle Competition](#task-1---kaggle-competition)
   - [Subtask 1.1 - Computer Vision](#subtask-11---computer-vision)
   - [Subtask 1.2 - NLP](#subtask-12---nlp)
2. [Task 2 - Underwater Image Enhancement Project](#task-2---underwater-image-enhancement-project)
   - [Subtask 2.1 - Variational AutoEncoders](#subtask-21---variational-autoencoders)
   - [Subtask 2.2 - GAN on MNIST](#subtask-22---gan-on-mnist)
   - [Subtask 2.3 - GAN on given Dataset](#subtask-23---gan-on-given-datasetn)
   - [Subtask 2.4 - Diffusion Model](#subtask-24---diffusion-model)
3. [Task 3 - RAG System](#task-3---rag-system)

## Task 1 - Kaggle Competition

This task focuses on building computer vision and NLP models on given datasets for a Kaggle Competition. Both of them have been completed successfully.

### Subtask 1.1 - Computer Vision

In this a CNN model has been built for DeepFake detection. One of the things to note in this project is that I assigned the wrong labels in the beginning for the images. Got to know this while submitting so to prevent training the model again I have used a simple logic to correct it after the models predict.
I have also used he initialisers to build the model seperately. I am specifically noting this because in the beginning i was getting a validation accuracy of only 50%. and the loss wasnt fluctuating during training. But on a random try it fluctuated and cal accuracy went upto 80% so I thought it might be due to weight initialisation so I next time I specifically used weight initaialisation techniques and as expected the accuracy increased,

### Subtask 1.2 - NLP

In this a Classifier model has been built using logistic regression. Things to note in this is that I have used GridsearchCV for hyperparameter tuning.

## Task 2 - Underwater Image Enhancement Project

The second task involves building different models for image enhancement and comparing them based on specific evaluation metrics.

### Subtask 2.1 - Variational AutoEncoders

In this a vaiational autoencoder model has been trained address image enhancement.

### Subtask 2.2 - GAN on MNIST

In this a GAN model has been implemented on MNIST dataset.

### Subtask 2.3 - GAN on given Dataset

In this a GAN model has been implemented on given Dataset. Thing to note in this subtask was that the loss function was specified for the task and a pixbypix loss function was used. It was interesting and fun to learn about this.

### Subtask 2.4 - Diffusion Model

In this a diffusion model was implemented for image enhancement. The base code for architecture was already provided.

## Task 3 - RAG System

In the final task a complete RAG system was implemented. Things to note in this was that it took me some time to learn about RAG. One of the major confusions I suffered was trying to access llama model through API key but it coming as gated repo with no access. Later I figured things out and used Ollama to pull the llama3 model on my local machine and then call it through langchain. This was a great project helped me learn a lot abt RAG, Langchain and AI Agents.

---

## Conclusion

Overall this was an enjoyable and interesting learning experience which helped dive in to many topics I had not yet explored. It also gave me the oppurtunity to learn a lot of things in just a week which if I had specifically tried to learn seperately might have taken a lot of time. One of the most important things was that it gave me an experience of practical coding and building of a model instead of just learning the theory. Thank you for this oppurtunity.
