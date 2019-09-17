---
layout: elements
title: 'K-means Clustering'
topic: 'Cluster Analysis'
weight: 1

asides:
  - The map is not the territory
---
{% include shortcut_variables.html %}
<div id="top-plot"></div>

You've just loaded a fresh dataset. It's still so hot from transformation that
your laptop fans haven't stopped whirring. After performing some basic
inspection, your eyes begin to glaze over and your face starts to go numb. Is
that your dead grandmother, Maureen? Wait, Maureen isn't dead, and she isn't
your grandma, she works in accounting. If you want to keep exploring the data,
you're going to need some patterns to hold on to before you pass out: market
segments in consumer data, tissues in medical images, communities in social
networks, topics in a text corpus. If only you could steady yourself on some
underlying subset structure lurking in your data, group them into different
categories according to the attribute values taken by related samples.

It's cluster hunting season.

Clusters are the truffles in your dataset, waiting to be found and finely grated
over your analysis. And like any good fungus hunter knows, the first
task is getting the right hog: one with a keen sense of smell, a lust tempered
by perspicacity, a snout for rooting but a maw you can muzzle. The second task
is not getting shot in the face by a paranoid French farmer. Ok, I may have
lost the thread of the analogy there, but the hogs we're talking about are
clustering methods. Very often, the first choice in cluster hogs is the
granddaddy of 'em all &mdash; k-means clustering.

Not only is k-means one of the oldest forms of cluster analysis, it is also one
of the most widely used. And it's the first approach to unsupervised learning
many budding datamancers encounter. K-means' enduring popularity both in
practice and pedagogy stems from the elegance in its simplicity. In my
experience, this unfortunately also means that k-means is one of the most
frequently misunderstood, poorly explained, and outright abused methods in
machine learning. Granted, my personal experience, like anyone's, is
statistically biased &mdash; no individual is an arbitrarily random sampler
&mdash; but I have seen enough incorrect implementations and misleading
characterizations to warrant address here. K-means may be elegant in its
simplicity, but that doesn't excuse simplicity in the minds of those who use it.

In this article I aim to impart an appreciation for the nuances of k-means
cluster analysis: what it is, how it works, and when it breaks.

One of the major sources of confusion about k-means can be traced to conflating
the problem with the algorithm used to solve it.

# The K-Means Problem

Let's go back to our hypothetical dataset. Hopefully your imaginary laptop's
fans have spun down by now. To make things more concrete, we'll say that we
have {{ils}}n{{ile}} data points, each of which has {{ils}}d{{ile}} features,
and each feature is a real number; more succinctly, each data
point {{ils}}x_{i}{{ile}} is a {{ils}}d{{ile}}-dimensional real vector
for {{ils}}1 \leq i \leq n{{ile}}.

Now suppose there *is*  some sort of meaningful, useful subset structure lurking
in the data, just waiting to be discovered. Before we can find it, we need to
decide how we will operationalize the distinguishing features of the subsets;
in other words, what measurable characteristics will we use to differentiate
one cluster from another?

## Means to an End
From a statistical point of view, an obvious choice to characterize
subpopulations is the average, or mean, of each subpopulation. Sample means are
easy to compute - just add and divide.
{{begin_aside}} Recall that a statistic is <dfn>sufficient</dfn> relative to an
unknown parameter of a particular statistical model if no other quantity
computed from a sample could provide any more information about the unknown
parameter. The notion of an <dfn>information bottleneck</dfn> is a related
information theoretic concept that quantifies how much information is lost about
the parameter of interest by using a particular statistic. In a certain sense,
the information bottleneck generalizes the concept of a sufficient statistic
from the perspective of lossy compression.
{{end_aside}}
Furthermore, depending on the true underling cluster structure and
what we plan to do with it, using only the means to characterize each cluster
could be sufficient in a statistical sense - the information bottleneck imposed
by using these single parameter cluster summaries could be immaterial to our
purposes.

An elegant weapon for a more civilized age. But now we have new issues to
address:
 - How many means do we need to estimate?
 - How are we going to estimate those means?

The 'k' in 'k-means' answers the first question. We assume we know how
many clusters there are <i class="latin">a priori</i>. Of course, this
assumption is artificial, as in practice if we knew how many clusters there were
beforehand we'd already know a lot about the underlying structure of our data.
{{begin_aside}} A similar complaint is often
raised by astute observers first learning about the Z-test: if we already know
the population variance, then we must also know the population mean since we've
measured the whole population. Similarly, knowing precisely how many clusters
are in our dataset would imply some significant knowledge of the underlying
subset structure of our dataset, which should make us question why we would need
to perform cluster analysis in the first place. {{end_aside}}
Estimating k when it is
unknown is a separate issue that we'll come back to later.

In order to deal with the second issue, we'll use a standard property of the
mean: it is the unique minimizer of the square error to all the sample points.
Formally, for a random sample of {{ils}}n{{ile}} independent
and identically distributed observations {{ils}} \{X_{i} \}_{i = 1}^{n}{{ile}},
the constant (vector) that uniquely minimizes the expected sum of the squared
errors to each random observation is the mean {{ils}}\mu{{ile}}:

<div>
$$
\DeclareMathOperator*{\argmin}{arg\,min}
\begin{align}
\mu = \argmin_{c \in \mathbb{R}^{d}} E\bigg[\sum_{i = 1}^{n}\vert\vert X_{i} - c \vert\vert_{2}^{2}\bigg]
\end{align}
$$
</div>
We want to apply this property of means not to the whole data set, but to
each of our {{ils}}k{{ile}} clusters individually. So for
the {{ils}}j^{th}{{ile}} cluster {{ils}}P_{j}{{ile}}, we
want to use just the mean {{ils}}\mu_{j}\) of that cluster.

<div>
$$
\mu_{j} = \argmin_{c \in \mathbb{R}^{d}} E\bigg[\sum_{X \in P_{j}}\vert\vert X - c \vert\vert_{2}^{2}\bigg]
$$
</div>

The main thing to notice here is that now our sum only runs over points in the {{ils}}j^{th}{{ile}}
cluster {{ils}}P_{j}{{ile}} rather than all {{ils}}n{{ile}} points.

We now have an objective function for each cluster. Since we don't know the
true mean {{ils}}\mu_{j}{{ile}} for each cluster {{ils}}P_{j}{{ile}},
we have to estimate it using our data. If your statistics skills can pay the
bills, you know that the sample average {{ils}}\bar{x}{{ile}} is an
unbiased estimator of the population mean and that,
<i class="latin">mutatis mutandis</i>, {{ils}}\bar{x}_{j}{{ile}}minimizes the
sum of square errors to all realized sample points in the {{ils}}j^{th}{{ile}}
cluster. So we just need to take the average of all the points in
cluster {{ils}}P_{j}{{ile}}, and we'll have our estimate
for {{ils}}\mu_{j}{{ile}} :
<div>
$$
\bar{x}_{j} = \argmin_{c \in \mathbb{R}^{d}} \sum_{x \in P_{j}}\vert\vert x - c \vert\vert_{2}^{2}
$$
$$
\text{where}\ \ \bar{x}_{j} = \frac{1}{|P_{j}|}\sum_{x \in P_{j}}x.
$$
</div>
But this isn't good enough - in order to estimate a cluster's mean using the
average of all data points belonging only to that cluster, we need to first know
which data points belong to that cluster. Of course, if we knew which points
belonged to each cluster, we wouldn't need to do any damn cluster analysis in
the first place! In order to address this dilemma, we incorporate the cluster
assignment information into our objective function: assign points to clusters
so that we minimize the *total within cluster*{: .emph} sum of square errors over *all possible
cluster assignments*{: .emph} {{ils}}\mathcal{P}{{ile}}. That is, out
of all partitions {{ils}}\mathcal{P}{{ile}} of our data
into {{ils}}k{{ile}} non-empty
subsets {{ils}}\mathbf{P} = \{P_{1}, P_{2}, \dots P_{k}\}{{ile}},
we want to find a particular partition {{ils}}\mathbf{P}_{0}{{ile}}
that minimizes the overall sum of squared errors,
<div>
$$\mathbf{P}_{0} = \argmin_{\mathbf{P}\in\mathcal{P}}\sum_{j = 1}^{k}\sum_{x \in P_{j}}\vert\vert x - \bar{x}_{j}\vert\vert_{2}^{2}.$$
</div>
Sounds simple enough, right? All we have to do is try out all the different ways
to split our data into {{ils}}k{{ile}} clusters, and find one that gives us the smallest
total when we add up the sum of square errors within each cluster.

## The Size of the Problem
Well I hope you've got a lot of free time, because the number of possible
partitions is *huge*. How huge? Mathematically speaking, it makes Dwayne Johnson
look like Kevin Hart - when he was five. The <dfn>Stirling numbers of the second
kind</dfn> count the number of ways to partition a set
of {{ils}}n{{ile}} objects
into {{ils}}k{{ile}} non-empty subsets.
For a given {{ils}}n{{ile}} and {{ils}}k{{ile}},
the corresponding Stirling number of the second kind,
denoted by {{ils}}n \brace k{{ile}}, is given by the
formula
{{begin_aside}}
The Stirling numbers of the second kind count the number of non-empty partitions
&mdash; every subset has to contain at least one point. This restriction is fine
for the k-means problem, but some algorithms that solve k-means may run into
issues of empty clusters.
{{end_aside}}
<div>
$$
{n \brace k} = \frac{1}{k!}\sum_{i=0}^{k}(-1)^{i}\binom{k}{i}(k - i)^{n}.
$$
</div>
See those factorials and binomial coefficients? Turns out we're dealing with
**super-exponential** growth. For {{ils}} n \geq 2{{ile}}
and {{ils}} 1 \leq k \leq n - 1 {{ile}}, we have the following lower and upper
bounds on these Stirling numbers:
<div>
$$
\frac{1}{2}(k^2 + k + 2)k^{n-k-1} - 1 \leq {n \brace k} \leq \frac{1}{2}\binom{n}{k}k^{n-k}.
$$
</div>

For a concrete example, if you have {{ils}}n = 100{{ile}} data points that you want to split
into {{ils}}k = 3 {{ile}} clusters, that means you have
<div>
$$
{100 \brace 3}  \approx 8.58 \times 10^{46}
$$
</div>
different clusterings to try out. It's estimated that there are on the order
of {{ils}} 10^{10} {{ile}} neurons in the average human cerebral cortex. Double
the size of the dataset, and now you've got
<div>
$$
{200 \brace 3} \approx 4.426 \times 10^{94}
$$
</div>
cluster configurations to search through. The number of baryons in the
observable universe is estimated to be on the order to {{ils}}10^{80}{{ile}}.
Forget watching paint dry, you may as well try to observe proton decay.

So, brute force is out. Any algorithm of practical use is going to need to be
more sophisticated than arm wrestling with time.

## Is the Mean Meaningful?
Implicit in the description of the k-means problem so far is that our data can
be modeled so that each cluster has a mean, and all {{ils}}k{{ile}} of the means
are distinct. In probabalistic terms, we have assumed that a mixture model
reasonably describes our data, with a one-to-one correspondence between mixture
components and clusters.
In reality, our data might have been generated by either a deterministic or
a stochastic process. Without delving into the thorny epistemological issues at
the bottom of science itself and human experience generally, I'm going to
suppose for the rest of this article that either a mixture model actually
generated our data, or that a mixture model describes the cluster structure of
our data arbitrarily well even though the true generating process is of some
other type. If this assumption were violated by our data set, then the k-means
model would have no relevance anyway.

We can say slightly more about the mixture model tacitly assumed by the k-means
problem: for each component, computing the mean is meaningful. What I mean is,
we have assumed both the existence and identifiability of each component
distribution's first moment.
<!--
Obviously, if
the population mean doesn't exist, the sample mean doesn't really capture any
valid description of the cluster. What's worse, if even one cluster has such a
pathological distribution, it can ruin our ability to describe the other
clusters. And even if the components all have finite first moments, if the tails
are fat enough that means 'barely' exist, unless we have a commensurately large
sample the sample means we compute are extremely likely to misrepresent the data.
While these concerns are usually paranoid delusions in practice, they cannot be
universally disregarded. Besides the Cauchy distribution, the Pareto
distribution does not have a finite expectation for certain parameter values.
And for other parameter values, the expectation 'barely' exists. The Pareto
distribution has been used to successfully model such varied processes as the
sizes of human settlements and hard disk error rates, so it is not just a
mathematical curiosity.
--->
# K-Means Algorithms

## Lloyd's Algorithm

The standard algorithm used to solve the k-means problem is often just called
'the k-means algorithm.' I find this name misleading for two reasons: it makes
it easier to confuse the problem posed with the method of solving it, and there
is actually more than one algorithm that can solve the problem. In order to
avoid both these issues, I'm going to use the original name from the computer
science literature: Lloyd's algorithm.

LLoyd's algorithm is an alternating, iterative procedure that shares structural
similarities with expectation maximization algorithms. The algorithm performs a
greedy optimization at each step in order to avoid examining every possible
partition of our data set. We can thus find an answer to our k-means problem
without the necessity of immortality &mdash; the trade-off is that the solution
produced is only guaranteed to be 'locally' optimal in a sense that we'll see
later.

The simplicity of Lloyd's algorithm is in large part responsible for the
ubiquity of k-means in cluster analysis. In it's most basic form, Lloyd's
algorithm is as follows:
{% include pseudocode.html id="Lloyd" code="
**Input:** positive integer k, dataset {x} \subset R^{d}
**Output:** k cluster Means
**Method:**
  Initialize \mu_{i} by randomly assigning k points in {x}.
  **Loop**
    *Assignment Step*
    Set P_{j} <- \empty for 1 \leq j \leq k
    **for** *i*=1, \dots, n:
      P_{j_{min}} <- P_{j_{min}} \cup x_{i}
      where j_{min} = argmin \vert\vert x_{i}-\mu_{j}\vert\vert _{2}^{2}
    *Estimation Step*
    **for** *j*=1, \dots k:
      \mu_{j} <- 1/\vert P_{j} \vert \sum_{x \in P_{j}} x
    **until** convergence"
%}
The greediness of the algorithm is apparent in both the assignment and
estimation steps: assign each point to the cluster mean that currently produces
the lowest square error, and then produce new estimates for each cluster mean
by using the assigned points. This procedure converges when there is no change
in the estimated cluster means, or equivalently, when there is no change in the
cluster assignments. No change in one implies no change in the other.

##

<script src="/assets/js/d3.js"></script>
<script src="/assets/js/elements/Unsupervised/Cluster_Analysis/kmeans.js"></script>
