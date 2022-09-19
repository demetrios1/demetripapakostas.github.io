---
abstract: |
  *\[Add abstract\]*
author:
  - |
    Andrew Herren\
    Demetrios Papakostas\
    P. Richard Hahn\
    Francisco Castillo
bibliography:
  - deep-causal-paper.bib
title: >-
  **Deep Learning for Causal Inference**\A Comparison of Architectures for
  Heterogeneous Treatment EffectEstimation
published: true
layout: post
subtitle: Each post also has a subtitle
gh-repo: daattali/beautiful-jekyll
gh-badge:
  - star
  - fork
  - follow
tags:
  - test
comments: true
---

# Introduction

A common problem in causal inference is to infer the effect of a binary
treatment, $$Z$$, on a scalar-valued outcome $$Y$$. When the effect of $Z$
  on $$Y$$ is posited to be constant for all subjects, or *homogeneous*, the
estimand of interest (average treatment effect) is a scalar-valued
parameter, which admits a number of common estimators. When the
assumption of treatment effect homogeneity is unwarranted, estimates of
the average treatment effect (ATE) may be of questionable utility. The
challenge of estimating a heterogeneous conditional average treatment
effect (CATE), is evident in the fact that the estimand is no longer a
scalar-valued parameter but a function of a (potentially high
                                             dimensional) covariate vector $X$. In recent years, researchers have
proposed to use machine learning methods for nonparametric CATE
estimation ([@hahn2020bayesian]; [@krantsevich2021stochastic],
            [@hill2011bayesian]; [@wager2018estimation]; [@farrell2020deep]).
Additional methods that have been introduced include TARNET [@shalit]
and Dragonnet [@shi2019adapting]. The main focus of this document will
be comparing [@farrell2020deep] and the method we introduce, as they are
the most similar in nature.

This paper focuses specifically on CATE estimators that rely on deep
neural networks. While neural networks are universal function
approximators ([@cybenko1989approximation]), nature does not typically
provide treatment effects for use as "training data,\" and estimation
proceeds by defining networks that can infer the CATE from available
data. The architecture of a deep neural network, which refers to a
specific composition of weights, data, and activation functions, plays a
crucial role in this process, along with regularization and training
techniques. This paper compares empirical CATE estimates of several
architectures. The first two methods represent outcomes as a sum of the
CATE, $\beta(X)$, and the prognostic effect $\alpha(X)$, which occurs
regardless of treatment status. In the [@farrell2020deep] architecture,
both $\alpha(X)$ and $\beta(X)$ emerge from a shared set of hidden
layers. Essentially, this architecture learns a common set of basis
functions for $\alpha(X)$ and $\beta(X)$ and then estimates separate
coefficients for each those basis functions. We refer to this approach
as "the Farrell method\" or simply "Farrell\" for the remainder of the
paper. The second method an extension of Bayesian Causal Forests (BCF)
([@hahn2020bayesian]). This method, which we hereafter refer to as
"BCF-nnet\" or "nnet BCF,\" uses completely separate neural networks for
$\alpha(X)$ and $\beta(X)$. Finally, we consider a "naive\" approach
that partitions the data into treatment and control groups, and learns a
separate function on each subset of the data. These functions can be
used to estimate the CATE by subtracting predictions of the "treatment
function" from those of the "control function.\"

Simulation studies show that nnet BCF outperforms both the Farrell and
naive methods when treatment effects are small relative to prognostic
effects.

# Problem Description

In order to make the problem precise, we begin by introducing notation
and defining our estimators. We will use the following conventions in
our notation:

-   Bold upper case letters (i.e. $$\mathbf{X}$$) refer to random matrices

-   Bold lower case letters (i.e. $$\mathbf{x}$$) refer to instantiations
    of random matrices

-   Regular upper case letters ($$X$$, $$Y$$, $$Z$$) refer to random variables
    / vector

-   Regular lower case letters ($$x$$, $$y$$, $$z$$) refer to instantiations
    of random variables

-   Math calligraphy letters ($$\mathcal{X}$$, $$\mathcal{Y}$$,
    $$\mathcal{Z}$$) refer to the support of a random variable

For example, if $$X \sim f(X)$$, we could write
$$\mathbb{E}\left(X\right) = \int_{\mathcal{X}} x f(x) \text{d}x$$.

Causal inference is concerned with the effect of a treatment, which we
denote as $$Z$$, on an outcome $$Y$$. In general, both the treatment and
outcome can be continuous, categorical, or binary. For the purposes of
this paper, we restrict our attention to the case of a binary treatment
($$\mathcal{Z} = \left\{0,1\right\}$$) and a continuous outcome
($$\mathcal{Y} = \mathbb{R}$$).

Our overview of the causal inference assumptions largely follows that of
[@hernan2020causal]. We are interested in inferring the effect of a
treatment, or intervention, on an outcome when nothing is changed except
the treatment being administered. In experimental settings, this causal
interpretation is often provided by the study design (randomized
controlled trial, randomized block design, etc\...). In many real-world
scenarios, designing and conducting an experiment would be impossible,
unethical, or highly impractical. In such cases, investigators are
limited to using observational data (i.e. data that were not collected
from a designed experiment).

To formalize the idea described above, we let $$Y^1$$ and $$Y^0$$ denote two
counterfactual random variables, where $$Y^i$$ indexes the random outcomes
for cases in which $$Z$$ has been set equal to $$i$$. The counterfactual
nature of these random variables is important to underscore before we
define assumptions and estimators. These two random variables are often
referred to as *potential outcomes* (see for example
[@hernan2020causal]). The variables are random because even in the
counterfactual scenario in which only treatment $i$ has been
administered, $$Y$$ may potentially be influenced by other factors.

We can define the average treatment effect (ATE) as
$$\beta = \mathbb{E}\left(Y^1 - Y^0\right) = \mathbb{E}\left(Y^1\right) - \mathbb{E}\left(Y^0\right)$$.
In many classical statistical problems, inferring the difference of two
random variables is as straightforward as assessing the difference in
the empirical means of samples of both random variables. However, the
counterfactual nature of $$Y^1$$ and $$Y^0$$ is such that for any
observation, only one of the two variables can be observed (subjects
cannot receive both the treatment and control). We define the observable
random variable $Y = ZY^1 + (1 - Z)Y^0$. In order to use a dataset of
independent and identically distributed (iid) samples of $$Z$$ and $$Y$$ to
estimate the ATE, we must make several *identifying* assumptions.

1.  *Exchangeability*: $$Y^1, Y^0 \perp Z$$

2.  *Positivity*: $$0 < P(Z = 1) < 1$$

3.  *Stable Unit Treatment Value Assumption (SUTVA)*:
    $$Y_i^1, Y_i^0 \perp Y_j^1, Y_j^0$$ for all $$j \neq i$$

With these assumptions, we can use observed data to estimate the average
treatment effect. One common estimator is the "inverse propensity
weighted\" (IPW) estimator, whose expectation is shown below to be the
ATE under the three assumptions of exchangeability, positivity, and
SUTVA. $$\begin{aligned}
        \frac{YZ}{p(Z = 1)} - \frac{Y(1-Z)}{1 - p(Z = 1)} &= \frac{\left(Y^1Z + Y^0(1 - Z)\right)Z}{p(Z = 1)} - \frac{\left(Y^1Z + Y^0(1 - Z)\right)(1-Z)}{1 - p(Z = 1)}\\
        &= \frac{Y^1Z^2 + Y^0(1 - Z)Z}{p(Z = 1)} - \frac{Y^1Z(1-Z) + Y^0(1 - Z)^2}{1 - p(Z = 1)}\\
        &= \frac{Y^1Z}{p(Z = 1)} - \frac{Y^0(1 - Z)}{1 - p(Z = 1)}\\
        \mathbb{E}\left(\frac{YZ}{p(Z = 1)} - \frac{Y(1-Z)}{1 - p(Z = 1)}\right) &= \mathbb{E}\left(\frac{YZ}{p(Z = 1)}\right) - \mathbb{E}\left(\frac{Y(1-Z)}{1 - p(Z = 1)}\right)\\
        &= \mathbb{E}\left(\frac{Y^1Z}{p(Z = 1)}\right) - \mathbb{E}\left(\frac{Y^0(1-Z)}{1 - p(Z = 1)}\right)\\
        &= \mathbb{E}\left(Y^1\right)\mathbb{E}\left(\frac{Z}{p(Z = 1)}\right) - \mathbb{E}\left(Y^0\right)\mathbb{E}\left(\frac{1-Z}{1 - p(Z = 1)}\right)\\
        &= \mathbb{E}\left(Y^1\right)\frac{p(Z = 1)}{p(Z = 1)} - \mathbb{E}\left(Y^0\right)\frac{1-p(Z = 1)}{1 - p(Z = 1)}\\
        &= \mathbb{E}\left(Y^1\right) - \mathbb{E}\left(Y^0\right)
    \end{aligned}$$ In practice, $P(Z = 1)$ is often estimated from the
data, but as long as $\mathbb{E}\left(\hat{p}(Z = 1)\right) = p(Z = 1)$,
the estimator will still be unbiased.

IPW is just one of many estimators of the average treatment effect. We
refer the interested reader to [@hernan2020causal] for more detail. We
now introduce the random variable $X$ to denote a vector of *covariates*
of the outcome (often referred to as "features\" in machine learning).
These covariates might include demographic variables, health markers
measured before treatment administration, survey variables measuring
attitudes or preferences, and so on.

Consider a simple motivating example, in which $$X$$ is age, $$Z$$ is a
blood pressure medication, and $$Y$$ is blood pressure. Older patients are
more likely to have high blood pressure, so we would expect that $$X$$ and
$$Y$$ are not independent. Older patients, who visit the doctor more
frequently, are also potentially more likely to be prescribed blood
pressure medicine. In this case, we would not expect $$Y^1, Y^0 \perp Z$$.
Older patients are more likely to receive blood pressure medicine and
also more likely to have high blood pressure so that observing $$Z = 1$$
changes the distribution of $$Y^1$$ and $$Y^0$$.

We can work around this limitation with a modified assumption,
*conditional exchangeability*: $$Y^1, Y^0 \perp Z \mid X$$. In words, this
states that, after we control for the effect of $X$ on treatment
assignment, the data satisfy exchangeability. Similarly, we no longer
have that $$P(Z = 1)$$ is the same for all subjects, so we modify the
positivity assumption to hold that $$0 < P(Z = 1 \mid X) < 1$$. Under this
set of assumptions, we define a new IPW estimator as
$$\frac{YZ}{p(Z = 1 \mid X)} - \frac{Y(1-Z)}{1 - p(Z = 1 \mid X)}$$ and
can show that its expected value is the ATE.

## Conditional Average Treatment Effect (CATE)

With the notation in place, we proceed to the focus of this paper:
estimating heterogeneous treatment effects using deep learning. In the
prior section, we introduced the average treatment effect as an expected
difference in potential outcomes across the entire support set of
covariates. Average treatment effects have a long history in the causal
inference literature because they are (relatively) straightforward to
estimate and provide useful, intuitive information about the average
benefits (or harms) of an intervention.

Sometimes, however, the ATE masks a considerable degree of heterogeneity
in the causal effects of an intervention. Consider the everyday example
of caffeine tolerance. Some people find that any level of caffeine
consumption at any time of day carries too many unpleasant effects,
while others drink espresso after a large dinner. While it may be
possible to measure an average treatment effect of a given dose of
caffeine, the estimate collapses a range of individual treatment effects
and may thus not provide much clinical or practical insight.

We define the *Conditional Average Treatment Effect* (CATE) as
$$\mathbb{E}\left(Y^1 - Y^0 \mid X = x\right)$$. Intuitively, this defines
a treatment effect for the conditional distribution of $$Y^1$$ and $$Y^0$$
in which $$X = x$$. Note that with this modification we define not a
single parameter $$\beta$$ but a function $$X \longrightarrow \beta(X)$$. If
$$X$$ is binary or categorical, this can be done empirically by
partitioning the data into subsets $$\left\{x: x = s\right\}$$ and then
estimating the ATE on the subsets. But in general, with continuous $$X$$
or simply a large number of categorical $$X$$ variables, this approach
becomes impossible and $$\beta(X)$$ must be estimated by fitting a model.

A tempting and convenient first step in CATE estimation would be use a
linear model for $$\beta(X)$$. More recently, advances in computer speed
and a growing recognition of the complexity of many causal processes has
spurred interest in nonparametric estimators of $$\beta(X)$$. To name a
few examples, [@hahn2020bayesian] and [@hill2011bayesian] use Bayesian
tree ensembles, [@wager2018estimation] use random forests, and
[@farrell2020deep] use deep learning. The focus of this paper will be to
compare the method introduced in [@farrell2020deep] to a novel
architecture inspired by [@hahn2020bayesian] and a naive partition-based
architecture.

## Estimating CATE using Deep Learning

We adapt the notation of [@farrell2020deep] slightly to fit the
conventions used above. As in prior sections, our goal here is to
estimate a causal effect of a binary treatment $$Z$$, on a continuous
outcome $Y$. Since we are interested in the effect's heterogeneity, we
must construct a model that will estimate
$$\mathbb{E}\left(Y^1 - Y^0 | X = x\right)$$ for any $$x$$. Before
discussing the specific architecture, we introduce some more clarifying
terminology and notation. This construction of treatment effect
heterogeneity follows that of [@hahn2020bayesian]. Consider the
following model $$\begin{aligned}
        Y &= \alpha\left(X\right) + \beta\left(X\right) Z + \varepsilon\\
        \varepsilon &\sim \mathcal{N}\left(0, \sigma_{\varepsilon} \right)\\
        Z &\sim \textrm{Bernoulli}\left(\pi(X)\right)
    \end{aligned}$$ In this case, $\beta\left(X\right)$ corresponds to
the treatment effect function, which given the assumptions in the prior
section, can be written as
$$\mathbb{E}\left(Y \mid X, Z = 1\right) - \mathbb{E}\left(Y \mid X, Z = 0\right)$$.
$$\alpha(X)$ corresponds to $\mathbb{E}\left(Y \mid X, Z = 0\right)$$
which we refer to as the *prognostic function*, $\varepsilon$ is random
noise, and $$\pi(X) = \mathbb{P}\left(Z = 1 | X\right)$$ which we refer to
as the *propensity function*.

Right now, $$X$$ refers to a (potentially large) vector of covariates that
may be useful in estimating heterogeneous treatment effects. But using
the above notation, we can partition $$X$$ into several categories:

1.  **Prognostic** features impact $$Y$$ through $$\alpha(X)$$

2.  **Effect-modifying** features impact the outcome $$Y$$ through
    $$\beta(X)$$

3.  **Propensity** features impact the outcome $$Y$$ through $$\pi(X)$$

For example if $$\pi(X) = \sin(X_1) + \abs{X_3}$$, we would say that $$X_1$$
and $$X_3$$ are propensity variables but $$X_2$$, for example, is not. These
categories are of course not mutually exclusive, but can be made so by
considering their combinations. We avoid the complete factorial
expansion of these three categories and instead define several
combinations that are of particular interest in methodological problems.

1.  **Pure prognostic** variables are variables which only appear in the
    function $$\alpha(X)$$

2.  **Pure modifiers** are variables which only appear in the function
    $$\beta(X)$$

3.  **Pure propensity** variables are variables which only appear in the
    function $$\pi(X)$$

4.  **Confounders** are variables which appear in both $$\pi(X)$$ and
    $$\alpha(X)$$

Before we proceed, we also introduce the concept of **targeted
selection**, when $$\pi(X) = f(\alpha(X))$$. Intuitively, this corresponds
to a practice of assigning treatment to those who are most likely to
need it (because, for example,
$$\alpha(X) = \mathbb{E}\left(Y | X, Z = 0\right)$$ would be high
otherwise). This is an extreme version of confounding, in which the
entire prognostic function is an input to the propensity function and
thus to the assignment of treatment. As is discussed in depth in
[@hahn2020bayesian], this phenomenon is both highly plausible in
real-world settings and also vexing to many approaches to CATE
estimation.

The architecture of the model is discussed in depth in later sections,
so here we simply note the high-level differences between the Farrell
method and nnet-BCF. [@farrell2020deep] propose to fit a model
$$\mathbb{E}\left(Y\right) = \alpha\left(X\right) + \beta\left(X\right) Z$$
using a neural network with two hidden layers to which map to two
separate output nodes: $$\alpha\left(X\right)$ and $\beta\left(X\right)$$.
[@hahn2020bayesian] fit a similar model using Bayesian Additive
Regression Trees (BART) ([@chipman2010bart]), with one key distinction.
$\alpha\left(X\right)$ and $\beta\left(X\right) Z$ are fit as completely
separate models with no information shared during training. This is
different from the [@farrell2020deep] approach as their method shares
weights between the $\alpha(X)$ and $\beta(X)$ functions via the first
two hidden layers. BCF nnet follows the approach of [@hahn2020bayesian]
by training two completely separate neural networks for $$\alpha(X)$$ and
$$\beta(X)$$. Finally, the "naive\" method estimates
$$\mathbb{E}\left(Y \mid X, Z = 1\right)$$ with one network and
$$\mathbb{E}\left(Y \mid X, Z = 0\right)$$ with another network so that
$$\beta(X)$$ can be estimated as a difference between these networks'
predictions.

# Methods

In this section, we discuss in more detail how the CATE is estimated in
each of the three deep learning methods proposed, as well as a linear
model comparison.

## Joint Training Architecture (Farrell/Shared Network)

In [@farrell2020deep], the authors posit that
$$\mathbb{E}\left(Y\mid X=x, Z=z\right)=G\left(\alpha(x)+\beta(x)z\right)
        \label{farrell_eq}$$ where $$G(u), u\in \mathbb{R}$$ is a known
link function specified by the researcher, and $$\alpha(\cdot)$$ and
$$\beta(\cdot)$$ are *unknown* functions to be estimated. Since we are
interested in effects of $$Z$$ on a real-valued $$Y$$, we use an identity
link function so that $$G()$$ can be removed from the equations and we
have $$\mathbb{E}\left(Y\mid X=x, Z=z\right)=\alpha(x)+\beta(x)z$$. The
authors propose estimating $$\alpha(\cdot)$$ and $$\beta(\cdot)$$ with one
deep fully connected neural network. We implement this architecture as a
fully connected neural network with two hidden layers and a two-node
parameter layer which outputs $$\alpha(X)$$ and $$\beta(X)$$. The output of
this architecture is then a linear combination of the two nodes in the
parameters layer, $$\alpha(x)+\beta(x)z$$ (see Figure
[1](#fig:farrell-pic){reference-type="ref"
reference="fig:farrell-pic"}).

![The Farrell method with a 3-dimensional vector of covariates $X$, 4
nodes in each hidden layer (in practice, these layers are usually much
deeper). $G$ is an activation function that takes $\alpha(X)+\beta(X)Z$
as an argument.](NN_draw_farrell_alpha){#fig:farrell-pic width="12cm"}

Since $$Y$$ is real-valued, we use mean squared error (MSE) as a loss
function in training each of the methods introduced in this section.

## BCF nnet

Based on the results and discussion in [@hahn2020bayesian], we
hypothesize that splitting $$\alpha(\cdot)$$ and $$\beta(\cdot)$$ into
separate networks with no shared weights may yield better CATE estimates
on some data generating processes (DGPs). The BCF nnet method specifies
$$\mathbb{E}\left(Y\mid X=x, Z=z\right)=\alpha\left(x, \hat{\pi}(x)\right)+\beta(x)z
        \label{fig:BCF_main}$$ In [@hahn2020bayesian], $\alpha$ and
$$\beta$$ are given independent BART priors ([@chipman2010bart]).
$$\hat{\pi}(x_i)$$ is an estimate of the propensity function We implement
the BCF nnet architecture as in Figure
[2](#fig:nnbcf-pic){reference-type="ref" reference="fig:nnbcf-pic"}.

![The BCF nnet architecture, where $$G(\cdot)$$ is an activation function
that takes $$\alpha(X)+\beta(X)Z$$ as an
argument.](NN_draw_alpha){#fig:nnbcf-pic width="16cm"}

While the shared-weights versus separate weights distinction between
Farrell and BCF nnet has been made clear, a subtle difference between
the architectures is that BCF nnet allows for an estimate of the
propensity function to be incorporated as a feature in the $$\alpha(X)$$
network. Since targeted selection implies $$\alpha(X)$$ is a function of
$$\pi(X)$$, this parameterization was observed to be helpful in
[@hahn2020bayesian].

In [@farrell2020deep], the authors develop confidence intervals for
their architecture's estimates (relying on influence functions, a common
tool for calculating standard errors in non-parametrics). We
incorporated these intervals into our architecture, but found that they
were far too tight and exhibited poor coverage in the low $n$ settings
we were studying. We therefore do not report or comment further.

## Separate Network Regression Approach

The "naive" method in our comparison employs two completely separate
regression models,
$$Y_1(X) = \mathbb{E}\left(Y \mid Z=1, X\right) \text{   and   } Y_0(X) = \mathbb{E}\left(Y \mid Z=0, X\right)
\label{eq:naive}$$ With these two regression functions, our estimate of
$$\beta(X)$$ is simply $$\beta(X) = Y_1(X) - Y_0(X)$$. Each $$Y_i(X)$$ is
constructed as a 2-layer fully connected neural network, with the number
of parameters chosen to be similar to the number chosen for the Farrell
and the BCF architecture.

## Linear Model

We also compare our two neural network architectures to a simple linear
model's estimate of $$\beta$$ $$Y=\beta Z+X\delta +\varepsilon
        \label{eq:linmod_eq}$$ where $$\beta$$ is the coefficient of
interest and represents the average treatment effect. The model is fit
using ordinary least squares (OLS). We allow for interaction effects
between $$X$$ and $$Z$$.

# Simulation Summary

is the first DGP we run. We choose a complex function for $$\alpha$$ and
strong targeted selection, and a simpler function for $$\beta$$ (which
allows for heterogeneous effects) to illustrate the effect of targeted
selection. $$\begin{aligned}
        \begin{split}
            X_1, X_2, X_3&\sim N(0,1)\\
            X_4&\sim \text{binomial}(n=2, p=0.5)\\
            X_5 &\sim \text{Bern}(p=0.5)\\
            X &= \left(X_1, X_2, X_3, X_4, X_5\right)\\
            \beta\left(X\right) &= \begin{cases}
                0.20+0.5*X_1\cdot X_4& \text{small treatment to prognosis}\\
                5+0.5*X_1\cdot X_4& \text{large treatment to prognosis}\\
            \end{cases}\\
            \alpha\left(X\right)&=0.5\cos\left(2X_1\right)+0.95*|X_3|\cdotX_5}-0.2*X_2+1.5\\
            \pi(X) &= 0.70*\Phi\left(\frac{\alpha(X)}{s(\alpha(X))}-3.5\right)+u/10+0.10\\
            u&\sim \text{uniform}(0,1)\\
            Y&= \alpha(X)+\beta(X)Z+\sigma\varepsilon\\
            \varepsilon &\sim N(0,1)\\
            \sigma  &= \text{sd}(\alpha(X))\cdot \kappa \\
            Z &\sim \text{Bern}(p=\pi(X))
        \end{split}
        \label{eq:dgp1}
    \end{aligned}$$

We choose the total number of parameters in the Shared architecture to
be about the same as the separate network ($$\alpha$$ + $$\beta$$ networks).
In the Shared network, this means we have 100 hidden nodes in layer 1,
and 26 in layer 2, meaning 3,280 total parameters.

In the BCF Nnet architecure, we have 60 parameters in the $$\alpha$$ first
layer, 32 hidden nodes in the second layer. For the $$\beta$$ network, we
have 30 and 20 hidden nodes respectively. This yields 3,226 total
parameters. For both methods, we use a learning rate of 0.001 with an
Adam Optimizer, we use Sigmoid activation, binary cross entropy loss for
the propensity, MSE for the other networks, ReLu activation (double
check), 250 epochs, and a batch size of 64. The dropout rate is 0.25 in
every layer. The propensity score for the BCF NNet architecture is
estimated using a 2 layer fully connected neural network with 100 and 25
hidden nodes respectively, and the rest of the parameters the same as
above. In the separate network approach, we build the architecture
separately using a 2-layer fully constructed neural network (for both
$$Y_1$$ and $$Y_0$$, as described in ) infrastructure with 50 hidden nodes
in layer 1 and 26 in layer 2 for both models. This is a total of 3,306
parameters. The other hyperparameters are the same as the BCF NNet and
Shared Network approach.

shows results using both the Shared Network approach [@farrell2020deep]
and the BCF Nnet approach we present. This table indicates some RIC
which biases the Farrell approach. The method we propose also has
additional flexibility in that the propensity estimate can be estimated
with any method and passed in, it need not be a MLP approach.
Additionally, because we separate the networks, like in the original BCF
paper [@hahn2020bayesian], we can add additional regularization on the
$$\beta$$ network[^1].

shows results with a large treatment to prognosis ratio. In this
setting, even with RIC presumably still being relevant due to the strong
targeted selection in (see right panel of ), the large treatment effect
dominating allows for the extra parameters of the shared network
approach to out-perform the separate network approach. However, as the
sample size increases, the gap disappears, leading us to believe with
sufficient sample size, this difference in methods would be minimal.

![Left panel: Histogram of $$\beta$$. On the right is a plot of $$\alpha$$
vs $$\pi$$, indicative of strong targeted selection. For this particular
realization of , with $$n=10,000$$, the mean of $$\beta(\bm{X})=0.20$$, the
mean of $$\alpha(\bm{X})=1.95$$, and the range of
$$\pi(\bm{X})=\qty(0.11, 0.90)$$, with mean of
0.37.](dgp_1_summary.png){#fig:dpg1 width="\\textwidth"}

![Left panel: Bias of dgp with different $$n$$. Right: RMSE. This is in
the "small" $$\beta$$ world.](summary_plot_smalltreat.pdf){#fig:BCF_better
width="\\textwidth"}

![Left panel: Bias of dgp with different $n$. Right: RMSE. This is in
the "small" $\beta$ world. COMMENT THIS OUT\...LIKE THE STYLE BUT
PROBABLY NOT APPROPRIATE](bias_plot_smalltreat.pdf){#fig:BCF_better
width="\\textwidth"}

![Left panel: Bias of dgp with different $n$. Right: RMSE. This is in
the "small" $$\beta$$ world. COMMENT THIS OUT\...LIKE THE STYLE BUT
PROBABLY NOT APPROPRIATE](rmse_plot_smalltreat.pdf){#fig:BCF_better
width="\\textwidth"}

![Comparing Individual biases and rmse's across the 100 Monte Carlo runs
for the shared and BCF architectures.
](compare_BCF_Farrell_indiv.pdf){#fig:bcf_vs_shared width="\\textwidth"}

# Data Example

-   Some ideas for data

-   [Glasses prevent covid](https://pubmed.ncbi.nlm.nih.gov/33910839/)I
    am trying to get access to the data that said eye-glass wearing
    reduced covid risk. Email request sent

-   Blue light glass effect on eyes and sleep, but there really isn't a
    lot of data out there on this

-   The Oregon medical lottery would be interesting, and we have access
    to that

-   Have access to driving fatality data by state and whether or not a
    ban on texting/cell phones is in place. These data are a combination
    of a few different datasets, so we have rich demographics as well as
    data on how many of the drivers involved were under the influence,
    but the time part makes the propensity function's role more iffy.
    Also, not entirely sure this is really confounded.

-   [Long covid](https://www.nature.com/articles/s41591-022-01909-w)
    This is also an interesting study that hopefully releases their data
    upon email request.

[@onyper2012class] is the data we look at for the moment. The dataset we
are interested in has 253 observations with 1 treatment variable, 1
outcome variable, and 11 variables defining our feature space $\bm{X}$.
Specifically, the data is collected on participating undergraduate
students at a liberal arts college in the northeastern United States.
Our outcome is \[`Poor Sleep Quality`\], which is a range of integers
from 1-18, with 1 being the worst sleep quality and 18 being the best
(the metric of interest is called the "Pittsburgh Sleep Quality Index"),
We standardize the data using the 'scale' command in R. The "treatment"
we investigate is \[`Stress Level`\], where students respond with "high"
or "normal". High stress level is considered the "treatment", and
presumably should lead to lower sleep quality. However, this problem is
a very clear causal inference problem, as illustrated in . It is not
unreasonable to expect there variables that could potentially lead to
higher stress *also* lead to worse sleep quality, and in particular
those described below. While this feature set likely does not fully meet
the strong ignorability assumption, we still muster on with the example.
The more worrying assumption violation is the SUTVA violation.

1.  `Gender`: Male/Female

2.  `Class Year`: Fresh, Soph, Jr, Sr.

3.  `Early Class`: Whether or not the student signed up for a class
    starting before 9AM

4.  `GPA`: College GPA, scale 0-4

5.  `Classes Missed`: Number of classes missed in semester

6.  `Anxiety Score`: Measure of degree of anxiety

7.  `Depression Score`: Measure of degree of anxiety

8.  `Happiness Score`: Measure of degree of happiness

9.  `Number Drinks`: Number of alcoholic drinks per week

10. `All Nighter`: Binary indicator for whether student had all-nighter
    that semester

11. `Average Sleep`: Average hours of sleep for all days

In our analysis, we change one hyper-parameter from our simulation
study. For the propensity estimation, we only run 100 epochs instead of
250 epochs. In the simulation, we found that more epochs led to better
predictive performance, but in the applied data problem, the propensity
estimate clustered to estimates of 0 and 1 for the probability. As a
benchmark, we compared our fully connected 2-layer neural network
estimate with a BART [@chipman2010bart] estimate, and found that with
100 epochs the estimates were similar. Otherwise, all parameters stayed
the same. Additionally, for this analysis we do not do a train/test
split.

![Left: A plot of $\alpha(\bm{X})$ vs $\pi(\bm{X})$, potentially
indicative of targeted selection. Of course, in this example the
selection is not targeted, but one could convince themselves people may
be more stressed out if they start thinking about sleep problems. Right:
A tree analysis, in the likes of
[@Woody2020].](beta_vs_pi.pdf){width="\\textwidth"}

![Left: A plot of $$\alpha(\bm{X})$$ vs $$\pi(\bm{X})$$, potentially
indicative of targeted selection. Of course, in this example the
selection is not targeted, but one could convince themselves people may
be more stressed out if they start thinking about sleep problems. Right:
A tree analysis, in the likes of
[@Woody2020].](tree_beta_fit.pdf){width="\\textwidth"}

# Discussion

What is preferrable about this proposed methodology to the state of the
art tree methods, such as [@wager2018estimation], [@hahn2020bayesian],
or [@krantsevich2021stochastic]? We do not aim to answer that question,
but rather instead provide some evidence that if a researcher is intent
on using some deep learning architecture for their causal needs, then
the methods developed in this document are the way to go. For one, we
show in a plausible simulation study the benefits of our methodology.
From a sheer performance point of view, our method provides an advantage
over other competitive deep learning causal tools. Additionally, the
parameterization of [@hahn2020bayesian] provides multiple advantages.
Because we split the prognosis and treatment networks, we can regularize
the networks differently, we could use different hyperparameters for
each, we can include different controls for each. This flexibility could
likely be of importance to practitioners with expert knowledge.
Additionally, because the approach is built off neural networks, it
lends itself to other applications, such as incorporating image data
into the control set (or even as a treatment). Building the network off
pytorch also allows for easier scalability, adjustment, and online
support.

# Acknowledgements {#acknowledgements .unnumbered}

We thank ASU Research Computing facilities for providing computing
resources.

[^1]: In the world of neural networks, this could entail changing
    dropout rates, implementing early stopping, or weight-decay, amongst
    other approaches. In general, an advantage of Neural Networks,
    particularly when using a well developed and maintained service like
    pyTorch is the ease in customizing one's model for one's needs.
