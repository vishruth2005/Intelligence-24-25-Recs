# Intelligence-24-25 Recruitment Submission

### By **Vishruth V Srivatsa** (231DS034)

---

This repository contains my completed recruitment task, divided into three main sections. Each task has been broken down into subtasks, with detailed instructions and outputs documented in the individual README files within the respective task directories.

---

## Table of Contents

1. [Task 1 - Kaggle Competition](#task-1---kaggle-competition)
   - [Subtask 1.1 - Computer Vision](#subtask-11---computer-vision)
   - [Subtask 1.2 - NLP](#subtask-12---nlp)
2. [Task 2 - Underwater Image Enhancement Project](#task-2---underwater-image-enhancement-project)

   - [Subtask 2.1 - Variational Autoencoders](#subtask-21---variational-autoencoders)
   - [Subtask 2.2 - GAN on MNIST](#subtask-22---gan-on-mnist)
   - [Subtask 2.3 - GAN on Given Dataset](#subtask-23---gan-on-given-dataset)
   - [Subtask 2.4 - Diffusion Model](#subtask-24---diffusion-model)

3. [Task 3 - RAG System](#task-3---rag-system)

---

## Task 1 - Kaggle Competition

This task involved building computer vision and NLP models on the provided datasets for a Kaggle competition. Both tasks were completed successfully.

### Subtask 1.1 - Computer Vision

I developed a CNN model for **DeepFake detection**. Initially, I incorrectly assigned labels to the images, but I corrected this with a simple logic after the model predictions, avoiding the need for retraining. I also focused on using **weight initialization** techniques. Initially, I was only achieving 50% validation accuracy, with no improvement in loss during training. However, after experimenting with weight initialization, the model's accuracy jumped to 95%, confirming the importance of initialization in this project.

### Subtask 1.2 - NLP

For the NLP subtask, I built a classifier model using **Logistic Regression**. I used **GridSearchCV** for hyperparameter tuning, optimizing the model for better performance.

---

## Task 2 - Underwater Image Enhancement Project

This task involved developing models for image enhancement, with a comparison of different approaches based on evaluation metrics.

### Subtask 2.1 - Variational Autoencoders (VAEs)

I trained a **Variational Autoencoder** (VAE) model to enhance underwater images, improving their clarity and quality.

### Subtask 2.2 - GAN on MNIST

I implemented a **Generative Adversarial Network (GAN)** model on the MNIST dataset, successfully generating new digit samples as part of this experiment.

### Subtask 2.3 - GAN on Given Dataset

For this subtask, I implemented a **GAN model on the provided dataset**. One key aspect of this project was working with a specified **pix2pix loss function**, which was fascinating to learn and apply.

### Subtask 2.4 - Diffusion Model

I implemented a **Diffusion Model** for image enhancement. The base architecture was provided, and I built on that to enhance underwater images.

---

## Task 3 - RAG System

In this task, I developed a complete **Retrieval-Augmented Generation (RAG) system**. One challenge I faced was accessing the **Llama model** through an API key, which required access to a gated repository. After resolving this issue, I pulled the **Llama3 model** locally using **Ollama** and integrated it with **Langchain**. This task was a deep dive into RAG systems, **Langchain**, and **AI agents**, which significantly enhanced my understanding of these technologies.

---

## Conclusion

This recruitment task provided an incredible learning experience, allowing me to dive into various new topics. I gained practical coding experience and developed real-world models in just a week—an opportunity that would have otherwise taken much longer if I had tackled these topics individually. It also highlighted the importance of applying theoretical knowledge in a hands-on setting.

Thank you for this opportunity!

---
