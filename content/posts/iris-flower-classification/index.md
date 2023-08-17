+++
draft = true
date = 2023-08-17
title = "Identifying Iris flowers using multi-class classification algorithm"
description = "using multi-class classification neural networks model to identify iris flower"
slug = ""
authors = ["Olumide Ogundele"]
tags = ["ml", "supervised learning", "classification algorithm"]
categories = ["machine learning"]
externalLink = ""
series = ["classification algorithm"]
+++

This is something I am so eager to write about because it's my first ever Machine Learning model.

After finishing my first Machine Learning (ML) specialization [Link] course on Coursera by Andrew Ng, I
wanted to reinforce the concepts of all I learned before starting Deep Learning specialization, so
this will be the first project in about 7 ML projects I need to do before I proceed with Deep Learning.

## The problem

### Identify irises

Irises influenced the design of the French fleur-de-lis, are commonly used in the Japanese art of flower arrangement known
as Ikebana, and underlie the floral scents of the “essence of violet” perfume [1]. They’re also the subject of this
well-known machine learning project, in which you must create an ML model capable of sorting irises based on five factors
into one of three classes, Iris Setosa, Iris Versicolour, and Iris Virginica.

To create a model capable of classifying each iris instance into the appropriate class based on four attributes:
sepal length, sepal width, petal length, and petal width.

### Choosing Machine Learning Algorithm

Looking at the problem at hand, it is a classification problem but in this case it's not a binary classification of a
yes or no, 0 or 1, since we can identify 3 flowers, it's a multi-class classification problem.

### Understand your data

When you can visualize, understand and know your data, it can give you more information about how to build your model,
see patterns that can help you make more informed decision in choosing your ML algorithm and also helps to know how to
transform the data in a way that can be passed as features (X) and targets (y) to the as learning data for the model
you are about to build.
