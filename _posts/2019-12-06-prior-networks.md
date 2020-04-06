---
layout: post
mathjax: true
title: "Uncertainty Estimation with Prior Networks"
date: 2019-11-30 23:59:40 +0100
categories: Uncertainty Estimation
---

## Introduction
The paper we aim to examine is titled [Predictive Uncertainty Estimation
via Prior Networks](https://papers.nips.cc/paper/7936-predictive-uncertainty-estimation-via-prior-networks.pdf) which was a conference paper at
NeurIPS 2018 in Montréal. The paper aims to solve the problem of
identifying sources of uncertainty by introducing a prior on the
predictive distribution. Uncertainty can come from various sources. What
we refer to as model uncertainty can also be called _epistemic
uncertainty_. This uncertainty stems from the failure of our model to
make accurate predictions and can be explained away given more data.
Another form of uncertainty is _aleatoric uncertainty_ or _data
uncertainty_ which is uncertainty stemming from noise that is inherent in
our data and cannot be explained. This can arise
for example from class overlap in a classification task. 

Historically,
to capture epistemic uncertainty one would place a prior distribution on
the model parameters while aleatoric uncertainty can be modelled by
placing a distribution on the model outputs.
In work done by Lakshminarayanan et al. [^4] and Kendall et
al. [^5], the two types of uncertainties are modeled
jointly. An ensemble is combined with variance estimates of the model
outputs to create a unified uncertainty measure. A problem with this
approach is that it can be of importance to separately identify the
sources of uncertainty in the model. For example, if uncertainty stems
from lack of data rather than inherent noise, an action to collect more
data can be taken.

# Prior Networks

The authors begin by establishing three rather than two separate sources
of uncertainty. Along with model uncertainty and data uncertainty the
authors also identify distributional uncertainty. Distributional
uncertainty is defined as the uncertainty that arises from a shift in
the distribution of the data, for example the difference between the
train and test set. This is different from past work mentioned where
distributional uncertainty is grouped together with either model
uncertainty or data uncertainty.

The uncertainties and the desired behaviour of our network is
illustrated in the figure below. We note that under high
data uncertainty, the network should
yield no class preference, but a certainty that the data is from the
in-distribution. This distinguishes itself from OOD samples which should
yield some measure of total uncertainty.

{% include image.html url="/assets/figures/dirichlet.png" description="Figure 1: Desired behaviour of an uncertainty network" %}

The uncertainty in a networks prediction for a classification task can
be modeled as 
$$\begin{equation}
    \label{eq:distribution}
    \tag{1}
    p(w_c |x^*, \mathcal{D}) = \int \underbrace{P(w_c|x^*, \theta)}_\text{Data}\underbrace{p(\theta|\mathcal{D})}_\text{Model}d\theta
    \end{equation}$$

Where $w_c$ denotes the class, $\theta$ the model parameters,
$\mathcal{D}$ the data and $x^{\*}$ a point drawn from the input data. By
marginalizing out the model parameters $\theta$, an expected
distribution of predictions is obtained. The issue with this approach is
that the posterior distribution over the weights $p(\theta|D)$ is often
intractable to compute. An approach to compute this measure is using a
variational approximation scheme [^8]. In the name of
simplicity, the authors estimate this distribution as
$$p(\theta|\mathcal{D}) = \delta(\theta - \hat{\theta})$$ To incorporate
distributional uncertainty into this model, the authors introduce the
term $p(\mu|x^*, \theta)$ which describes the distribution over
predictive categoricals. Data uncertainty is then modeled as
$P(w_c|\mu)$ where $\mu$ is the parameters of a categorical
distribution. Equation $\ref{eq:distribution}$ is thus modified as

$$\begin{equation}
\label{eq:prior_dist}
\tag{2}
    p(w_c |x^*, \mathcal{D}) = \int \underbrace{P(w_c|\mu)}_\text{Data}\underbrace{p(\mu|x^*, \theta)}_\text{Distributional}\underbrace{p(\theta|\mathcal{D})}_\text{Model}d\mu d\theta
    \end{equation}$$

Equation $\ref{eq:prior_dist}$ induces a hierarchy of uncertainties
where model uncertainty affects distributional uncertainty which affects
estimates of data uncertainty. Further measures of uncertainty can then
be found by marginalizing particular variables. For example by
marginalizing over $\mu$ we find Equation $\ref{eq:distribution}$.

Dirichlet Prior Networks
------------------------

In practice, we model $p(\mu|x^{\*}, \theta)$ as a Dirichlet distribution
with parameters $\alpha = f(x^*; \theta)$ where $f$ is our model. This
acts as a prior on the categorical distribution over class labels
$p(w_c|\mu)$. Now as 

$$\begin{aligned}
    p(w_c|\mu) &= \mu_c \\
    \int p(w_c|\mu)p(\mu|x*, \theta) d\mu &= \int u_i p(\mu|\alpha) d\mu = \frac{\alpha_i}{\sum_i \alpha_i}
    \end{aligned}$$


we see that the posterior distribution over class labels,
$p(w_c|x^{\*}, \theta)$, will be given by the expected value of the
Dirichlet distribution. To ensure that the network outputs are strictly
positive we let $f(x^*, \theta) = (z_c)_c$ and let $\alpha_c = e^{z_c}$.
To train the model, a certain objective function is designed to
facilitate two desired behaviours of the network. 

$$\begin{equation}
    \label{eq:loss}\tag{4}
    \mathcal{L}(\theta) = \mathbb{E}_{p_{in}(x)}[KL[Dir(\mu|\hat{\alpha})||p(\mu|x,\theta)]] + \mathbb{E}_{p_{out}(x)}[KL[Dir(\mu|\tilde{\alpha})||p(\mu|x,\theta)]]
    \end{equation}$$

The loss function is designed as a multi-task objective on the Dirichlet
prior distribution. The distributions $p_{in}$ and $p_{out}$ represent
the distribution of $x$ on the training data and data sampled from an
out-of-distribution dataset. One one hand, data sampled from the
out-distribution should capture the behaviour of a flat Dirichlet
distribution with $\tilde{\alpha_c} = 1$, to ensure that no specific
class is preferred over another. On the other hand, the in-distribution
data should be modeled as the target data. To set $\hat{\alpha}$, we let

$$
\begin{equation}
\hat{\alpha_c} = \hat{\mu}_c (\sum_{i}^n \hat{\alpha}_i)
\end{equation}
$$

 where
$\hat{\mu}_c$ is the one-hot binary targets to some modification [^1].
Finally, the Kullback-Liebler Divergence is used as a distance metric
between these distributions which combined gives our loss function.

Experiments and Results
=======================

A number of measures are used to evaluate the performance of the
uncertainty estimation. We define them in the table below.

  ---------------------------- ------------------------------------------------
  Maximum class probability    $\mathcal{P} = max_c P(w_c |x^*; \mathcal{D})$
  Entropy                      $\mathcal{H}[P(y|x^*;D)]$
  Model Uncertainty            $\mathcal{I}[y,\theta|x^*, \mathcal{D}]$
  Distributional Uncertainty   $\mathcal{I}[y,\mu|x^*, \mathcal{D}]$
  Differential Entropy         $\mathcal{H}[P(\mu|x^*;D)]$
  ---------------------------- ------------------------------------------------

  : Uncertainty estimates used to evaluate the
  model

To compare measures of uncertainty, a toy dataset is constructed by
drawing samples from three Gaussian distributions with equidistant means
and equal variance. By altering the variance we get differing degrees of
class overlap. The experiments illustrates the difference between data-
and distributional uncertainty. The uncertainty arising from samples
taken outside of our three Gaussians represents out-of-distribution
samples while samples that exist in the class overlap would be an
example of data uncertainty. The experiment is replicated using PyTorch to get a more thorough understanding of the model. [^2]


{% include image.html url="/assets/figures/prior_entropy.png" description="Figure 2: Entropy plots from the model trained with data sampled from two
different mixtures of gaussians with different levels of overlapping
classes. In the top row, as the class overlap is non-existant, the
entropy and differential entropy behave the same. As you increase the
class overlap, the entropy is still high in the class overlap while the
differential entropy is
low." %}

In the figure above, the Entropy of the posterior distribution
over class labels $\mathcal{H}[p(w_c|x^{\*}, \theta)]$ and the differential
entropy $\mathcal{H}[p(\mu|x*, \theta)]$ are plotted together with a
visualization of the input data. There are two types of uncertainties
present in the Figure. Points from outside of our Gaussians represents
OOD data and give rise to distributional uncertainty while data in the
class overlap would give rise to data uncertainty. Entropy is high in
both OOD regions and in regions of class overlap. This illustrates it's
difficulty to distinguish distributional and data uncertainty. In
contrast, the Differential Entropy is low in regions of class overlap.
This behaviour can be linked to the fact that Differential Entropy
measures the uncertainty in the distribution. Regions of class overlap
still belong the the training distribution and hence the differential
entropy is low while the entropy is high since it cannot distinguish
between classes in these regions.
Further experiments are presented in the paper where the uncertainty
measures are used to both identify misclassification in in-distribution
data and for out-of-distribution input detection. The experiments are
run on the MNIST, CIFAR-10 and Omniglot datasets among others. The experiments use the uncertainty
measures to distinguish between classes. The area under the ROC (AUROC)
and Precision-Recall (AUPR) curves are used to judge performance since
no fixed threshold of what level of uncertainty determines class
membership is set. The results of the Dirichlet Prior Network are
compared to those of the class posterior of a DNN[^6] and an
MC dropout ensemble [^7]. The results are shown in the table below.

{% include image.html url="/assets/figures/table2_paper.png" description="Results of the OOD experiment" %}

We note that the DPN model outperforms the baseline models under all
measures and dataset compositions. The authors make a note of the high
performance against the TinyImageNet dataset since this
dataset is of a more similar nature to CIFAR-10 and thus provides a more
difficult learning task.

Conclusion
==========

The paper provides a novel method for uncertainty estimation. The main
contribution is successfully incorporating different uncertainty
measures into the model and realizing the importance of distinguishing
data uncertainty from distributional uncertainty. Through
re-implementing some of the experiments, a few questions arise regarding
hyperparameter choice and experiment design. In the toy experiment shown
in Figure 2, the conclusion drawn by the authors is that
differential entropy can better identify distributional uncertainty. It
would be interesting for the authors to clarify their choice of
hyper-parameters, specifically the choice of precision for the
in-distribution Dirichlet prior in Equation ($\ref{eq:loss}$). I found that
by increasing this parameter, the Differential Entropy increased in
regions of class overlap and thus making it indistinguishable from the
Entropy. It would be fruitful for the authors to comment on this
phenomenon. Furthermore, I found it unclear in all experiments which
data was used to act as OOD data during training. For example, the
out-of-domain detection experiment specifies both the in-distribution
and out-of-distribution datasets. Does this imply that it was trained
using these two datasets, or that it was trained using the ID dataset
together with a random noise OOD dataset and then evaluated on the
specified OOD dataset? In the first case, the model provides a firm
upperhand against the baselines since it's specifically trained to
provide a flat Dirichlet prior on the OOD samples while in the second
case, a more detailed explanation of the training procedure could be
provided in an Appendix.


While experiment details are vague, the strengths of the paper lie in
its clear exposition of the theory behind its main ideas and it provides
an intuitive account for why uncertainty is important to estimate and
why current approaches are lacking. The Dirichlet prior has been well
studied in Bayesian Learning, yet the connection to how one incorporates
it into a neural network is novel. I think the paper highlights an
important aspect in how prior knowledge of a problem is incorporated
into a machine learning model. RNNs and CNNs are really architectural
choices that have been chosen from prior beliefs on how to model
sequential and visual tasks. Prior Networks have no special
architecture, the model is a simple feed-forward network with Softmax
outputs. The prior knowledge is instead entirely captured by the loss
function which comes about by cleverly reinterpreting the outputs of a
Softmax-classifier as the mean of the Dirichlet distribution.
Furthermore, the specific design of the loss is very intuitive given the
background introduction to how an ideal model of uncertainty would
behave. It is also impressive how in this
way, the authors circumvents the problem of placing a complex prior on
the model weights as in previous approaches.


Overall, the paper provides an interesting idea and impressive results
compared to past work, however some details of the experiments could be
expanded upon. As the authors note, it would be interesting to extend
this work for regression tasks. Another extension is exploring different
loss functions. In fact, the authors have an accepted paper to NeurIPS
2019 where they explore reverse KL-divergence training on Prior Networks
[^3]. The main takeaways of the paper is that there is still
much to find in the intersection between deep learning and Bayesian
statistics. Deep Learning works well, to understand why and to make it
better, a new architecture or optimizer may not be needed. Instead,
looking at existing methods through a Bayesian lens can give invaluable
insight.

[^1]: Details in original paper Eq. 13

[^2]: Code available at <https://github.com/gtegner/PriorNetworks>

[^3]: <https://papers.nips.cc/paper/9597-reverse-kl-divergence-training-of-prior-networks-improved-uncertainty-and-adversarial-robustness.pdf>
 
[^4]: <https://arxiv.org/pdf/1612.01474.pdf> "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"

[^5]: <https://arxiv.org/pdf/1703.04977.pdf> "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"

[^6]: <https://arxiv.org/pdf/1610.02136.pdf> "A baseline for detecting misclassified and Out-of-Distribution examples in Neural Networks"

[^7]: <https://arxiv.org/pdf/1506.02142.pdf> "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"

[^8]: <https://arxiv.org/pdf/1505.05424.pdf> "Weight Uncertainty in Neural Networks"