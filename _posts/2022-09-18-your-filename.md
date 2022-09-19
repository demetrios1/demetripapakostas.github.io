---
published: false
---
## A New Post

Enter text in [Markdown](http://daringfireball.net/projects/markdown/). Use the toolbar above, or click the **?** button for formatting help.

## Introduction
	A common problem in causal inference is to infer the effect of a binary treatment, 
	$Z$, on a scalar-valued outcome $Y$. When the effect of $Z$ on $Y$ is posited to be constant 
	for all subjects, or \textit{homogeneous}, the estimand of interest (average treatment effect) is a scalar-valued 
	parameter, which admits a number of common estimators. When the assumption of treatment effect homogeneity is 
	unwarranted, estimates of the average treatment effect (ATE) may be of questionable utility. The challenge of 
	estimating a heterogeneous conditional average treatment effect (CATE), is evident in the fact that the estimand is no longer 
	a scalar-valued parameter but a function of a (potentially high dimensional) covariate vector $X$. In recent years, 
	researchers have proposed to use machine learning methods for nonparametric CATE estimation (\cite{hahn2020bayesian}; \cite{krantsevich2021stochastic},
	\cite{hill2011bayesian}; \cite{wager2018estimation}; \cite{farrell2020deep}).  Additional methods that have been introduced include TARNET \cite{shalit} and Dragonnet \cite{shi2019adapting}.  The main focus of this document will be comparing \cite{farrell2020deep} and the method we introduce, as they are the most similar in nature.  
	
	This paper focuses specifically on CATE estimators that rely on deep neural networks. 
	While neural networks are universal function approximators (\cite{cybenko1989approximation}), 
	nature does not typically provide treatment effects for use as ``training data," and estimation 
	proceeds by defining networks that can infer the CATE from available data.
	The architecture of a deep neural network, which refers to a specific composition of weights, data, and 
	activation functions, plays a crucial role in this process, along with regularization and training techniques. 
	This paper compares empirical CATE estimates of several architectures. The first two methods represent 
	outcomes as a sum of the CATE, $\beta(X)$, and the prognostic effect $\alpha(X)$, which occurs regardless 
	of treatment status. In the \cite{farrell2020deep} architecture, both $\alpha(X)$ and $\beta(X)$ emerge from a 
	shared set of hidden layers. Essentially, this architecture learns a common set of basis functions for 
	$\alpha(X)$ and $\beta(X)$ and then estimates separate coefficients for each those basis functions.
	We refer to this approach as ``the Farrell method" or simply ``Farrell" for the remainder of the paper.
	The second method an extension of Bayesian Causal Forests (BCF) (\cite{hahn2020bayesian}). This method, 
	which we hereafter refer to as ``BCF-nnet" or ``nnet BCF," uses completely separate neural networks for 
	$\alpha(X)$ and $\beta(X)$. Finally, we consider a ``naive" approach that partitions the data into 
	treatment and control groups, and learns a separate function on each subset of the data. These functions can be 
	used to estimate the CATE by subtracting predictions of the ``treatment function'' from those of the ``control function."
