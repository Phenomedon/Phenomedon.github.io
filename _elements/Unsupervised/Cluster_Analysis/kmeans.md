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
is not getting shot in the face by a paranoid French farmer [^1]. Ok, I may have
lost the thread of the analogy there, but the hogs we're talking about are
clustering methods. Very often, the first choice in cluster hogs is the
granddaddy of 'em all &mdash; k-means clustering.

Not only is k-means one of the oldest forms of cluster analysis, it is also one
of the most widely used. And it's the first approach to unsupervised learning
many budding datamancers encounter. K-means' enduring popularity both in
practice and pedagogy stems from its elegance and simplicity. In my
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
{: .section}

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
{: .subsection}

From a statistical point of view, an obvious choice to characterize
subpopulations is the average, or mean, of each subpopulation. Sample means are
easy to compute - just add and divide.
{{begin_aside}} Recall that a statistic is <dfn>sufficient</dfn> relative to an
unknown parameter of a particular statistical model if no other quantity
computed from a sample could provide any more information about the unknown
parameter. The notion of an <dfn>information bottleneck</dfn> is a related
information theoretic concept that quantifies how much information is lost about
the parameter of interest by using a particular statistic. In a certain
sense, the information bottleneck generalizes the concept of a sufficient
statistic from the perspective of lossy compression.
{{end_aside}}
Furthermore, depending on the true underling cluster structure and
what we plan to do with it, using only the means to characterize each cluster
could be sufficient in a statistical sense - the information bottleneck [^2] imposed
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
{{begin_aside}}
While the sample mean is the unique minimizer of the square error to all sample
points, {{ils}}\sum_{x \in P_{j}}\vert\vert x - \bar{x}\vert\vert{{ile}}, and an
unbiased estimator of the population mean, {{ils}}E[\bar{x}]=\mu_{j}{{ile}}, it is
not necessarily the unbiased estimator of the population mean with minimum
variance, {{ils}}\bar{x}\overset{?}{=} \argmin_{c \in \mathbb{R}^{d}}E[\vert\vert \mu_{j} - c\vert\vert_{2}^{2}]{{ile}}.
<br>
If the population distribution is Gaussian, for example, then the sample mean is
the MVUE for the population mean. In other cases, such as when the distribution
is uniform over an unknown range {{ils}}[a, b]{{ile}}, the sample mean is <em>not</em>
the MVUE for the population mean. This perhaps subtle point becomes more
relevant when considering assumptions and properties of solutions produced by
the algorithms discussed in the next section.
{{end_aside}}
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
$$
\mathbf{P}_{0} = \argmin_{\mathbf{P}\in\mathcal{P}}\sum_{j = 1}^{k}\sum_{x \in P_{j}}\vert\vert x - \bar{x}_{j}\vert\vert_{2}^{2}. \tag{1}\label{1}
$$
</div>
{{begin_aside}}
<strong>A quick note on terminology</strong> &mdash; for a cluster {{ils}}P_{j}{{ile}}, the
quantity {{ils}}\small\sum_{x \in P_{j}}\vert\vert x - \bar{x}_{j}\vert\vert_{2}^{2}{{ile}} is
commonly reffered to as the <dfn>Within Cluster Sum of Square Errors</dfn> (WCSS).
The <dfn>Within Cluster Variance</dfn>,  {{ils}}\small\text{Var}(P_{j}){{ile}}, is the familiar sample variance formula
applied to the cluster, {{ils}}\small\frac{1}{\vert P_{j}\vert}\sum_{x \in P_{j}}\vert\vert x - \bar{x}_{j}\vert\vert_{2}^{2}{{ile}}. The
only difference between these two quantities is the normalization factor {{ils}}\small\frac{1}{\vert P_{j}\vert}{{ile}}.
Except in cases where the distinction is necessary, I will adopt a common abuse
of terminology and consider the WCSS as synonymous with the cluster variance,
forgetting about the normalization.
Since the mean minimizes both for a fixed cluster, which you can see for
yourself with some elementary calculus, the distinction is mostly irrelevant to
our purposes here.
{{end_aside}}
Sounds simple enough, right? All we have to do is try out all the different ways
to split our data into {{ils}}k{{ile}} clusters, and find one that gives us the
smallest total when we add up the sum of square errors within each cluster.

## The Size of the Problem
{: .subsection}

Well I hope you've got a lot of free time, because the number of possible
partitions is *huge*. How huge? Mathematically speaking, it makes Dwayne Johnson
look like Kevin Hart - when he was five. The <dfn>Stirling numbers of the second
kind</dfn> count the number of ways to partition a set
of {{ils}}n{{ile}} objects
into {{ils}}k{{ile}} non-empty subsets.
For a given {{ils}}n{{ile}} and {{ils}}k{{ile}},
the corresponding Stirling number of the second kind,
denoted by {{ils}}n \brace k{{ile}}, is given by the
formula [^3]
<div>
$$
{n \brace k} = \frac{1}{k!}\sum_{i=0}^{k}(-1)^{i}\binom{k}{i}(k - i)^{n}. \tag{2}\label{2}
$$
</div>
{{begin_aside}}
The Stirling numbers of the second kind count the number of non-empty partitions
&mdash; every subset has to contain at least one point. This restriction is fine
for the k-means problem, but some algorithms that solve k-means may run into
issues of empty clusters.
{{end_aside}}
See those factorials and binomial coefficients? Turns out we're dealing with
**exponential** growth. For {{ils}} n \geq 2{{ile}}
and {{ils}} 1 \leq k \leq n - 1 {{ile}}, we have the following lower and upper
bounds on these Stirling numbers [^4]:
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
{: .subsection}

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
problem: for each component, computing the mean is meaningful. What I mean is
(I'll stop now, I mean it), we have assumed both the existence and
identifiability of each component distribution's first moment.
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
{: .section}

Since the space of possible solutions is generally exponential in the
cluster number and dataset size, all commonly used algorithms applied to the
k-means problem are heuristic in nature.

## Lloyd's Algorithm
{: .subsection}

The standard algorithm used to solve the k-means problem is often just called
'the k-means algorithm.' I find this name misleading for two reasons: it
promotes conflation of the problem posed with the method of solving it, and
there are actually several algorithms that can solve the problem. In order to
avoid both these issues, I'm going to use the original name from the computer
science and pattern recognition communities: Lloyd's algorithm [^5].
Stuart Lloyd of Bell Laboratories originally proposed the procedure in 1957 for
the  purpose of finding the minimal distortion signal quantization in the
context of pulse code modulation [^6].

Unless you're familiar with signal processing, that last sentence probably
sounds like some nonsense J.J. Abrams used as dialogue in a Star Wars movie.
Basically, Lloyd was working on transmitting digital signals over a noisy
channel. The receiver stored {{ils}}k{{ile}} representatives, and each
transmitted sample, corrupted by noise over the channel, had to be assigned to
one of the {{ils}}k{{ile}} representatives. Lloyd offered a solution using
the specific {{ils}}k{{ile}} representatives that minimized the distortion
between the original signal and the version reconstructed by the receiver.
Distortion in this case was measured by the power in the difference between the
signals &mdash; and this power is just the average square of the noise. So Lloyd
needed to minimize square error, and historically, we can see why knowing the
value of {{ils}}k{{ile}} in advance wasn't so outlandish in the context of
signal recovery.

LLoyd's algorithm is an alternating, iterative procedure that shares structural
similarities with expectation maximization algorithms. The algorithm performs a
greedy optimization at each step in order to avoid examining every possible
partition of our data set. We can thus find an answer to our k-means problem
without the necessity of immortality &mdash; the trade-off is that the solution
produced is only guaranteed to be 'locally' optimal in a sense that we'll make
concrete later.

The simplicity of Lloyd's algorithm is in large part responsible for the
ubiquity of k-means in cluster analysis. In it's simplest form, Lloyd's
algorithm is as follows:

<div>
  <img width="1fr" class="svg-pseudo"
  src="/assets/images/elements/Unsupervised/Cluster_Analysis/lloyd_pseudocode.svg"
  onload="SVGInject(this)"/>
</div>

The pseudocode above focuses on the main idea of the algorithm, and thus certain
details covering edge cases and efficient implementation are left out. In lines
2 through 7, we select as initial centroids {{ils}}k{{ile}} data points
uniformly at random. The use of the uniform distribution should be augmented by
a procedure that ensures each data point selected as an initial centroid is
distinct. Distinct initial centroids can be guaranteed by a standard rejection
sampling scheme. The main loop comprises the rest of the algorithm in
lines 8 through 21, and alternates between two main steps.

The first for loop in lines 11 through 14 is called the assignment step,
and assigns each data point {{ils}}x_{i}{{ile}} to the cluster with the centroid
that produces the smallest square error. The assignments are indicated by the
values of the {{ils}}r_{i}{{ile}}, where we have {{ils}}\forall\ i, r_{i} \in \{1, \dots, k\}{{ile}}.

The second for loop in lines 15 through 18 is called the update or estimation
step, and consists of computing new estimates for each cluster's centroid by
averaging all the points assigned to that cluster in the immediately preceding
assignment step.

The main loop terminates when all centroid estimates are unchanged from one
iteration to the next.

### Forgy Initialization
{: .subsubsection}
The initialization method in Lloyd's algorithm deserves more attention. In the
above procedure, the initial centroids are selected from the input data
uniformly at random. This initialization procedure is often attributed to
E.W.&nbsp;Forgy [^7], and the overall iterative method is sometimes called the
Lloyd-&nbsp;Forgy algorithm: for instance, in the kmeans function of the stats
package in R. There is no theoretical justification for such an
initialization, and it can in fact produce undesirable results.

Ideally, the initial centroids will each be selected from *distinct* clusters.
Such an ideal initialization would produce a much better preliminary
representation of the data's cluster structure, and almost certainly a final
clustering that accurately captures the target patterns. Of course, guaranteeing
such a representative initialization would require knowing the cluster
structure, and we are again left only with heuristic methods.

For a dataset with {{ils}}n \gg k{{ile}}, selecting the initial centroids
independently may cause some clusters to be unrepresented. Because Lloyd's
algorithm is only guaranteed to find a local optimum, such a poor starting
representation produced by Forgy initialization can significantly degrade
performance.

Alternatively, we may determine the initial centroids by:
- Initially assigning each point to a cluster uniformly at random. In other
words, for {{ils}}1 \leq i \leq n{{ile}}, select {{ils}}r_{i} \sim \scriptsize{\text{UNIFORM}}\{1, k\} {{ile}}.
- Compute each initial centroid as we would during a typical estimation step,
i.e., set {{ils}}\mu_{j}^{\scriptsize{(0)}} = \frac{1}{\vert P_{j}\vert}\scriptsize{\displaystyle\sum_{x \in P_{j}}}\hspace{0.2em} x {{ile}}.

The above initialization method is often referred to as Random Partition
initialization, although there are conflicting claims as to who originated
each of these initialization methods [^7][^8][^9].

Some passing thought should make apparent that the initial
centroids produced by random partition will tend to be near the collective
centroid of the entire dataset. Such an initialization can lead to its own
problems, and is as arbitrary a starting point as Forgy initialization.
{{begin_aside}}
Some authors attribute what I have called Forgy initialization here to MacQueen
<sup id="fnref:5:a"><a href="#fn:5" class="footnote">5</a></sup>
<sup id="fnref:9:0"><a href="#fn:9" class="footnote">9</a></sup>, others to
McRae <sup id="fnref:8:a"><a href="#fn:8" class="footnote">8</a></sup>,
and instead credit Forgy with the Random Partition method. These conflicting
claims of origination can not be arbitrated by primary sources, as Forgy's
initial presentation was given as a conference talk, which is apparently only
described in secondary sources <sup id="fnref:5:b"><a href="#fn:5" class="footnote">5</a></sup>
such as <sup id="fnref:8:b"><a href="#fn:8" class="footnote">8</a></sup>.
<br/>
Regardless of who should be credited with either of these initialization
methods, they are both equally crappy: three plunger worthy. The typical
recommendation for dealing with such poor initialization methods is to simply
perform them so many times that some coherence is achieved in the resulting
clusters. I think a better approach is to use more principled initialization
methods, which are discussed in a following section.
{{end_aside}}

Extensive empirical results on real and synthetic data [^9] demonstrate that
both Forgy and Random Partition initialization are among the worst methods
to start Lloyd's algorithm. For even reasonably small datasets, Forgy
initialization tends to vary wildly from one run to the next, and, as discussed
above, may under represent the cluster structure by selecting initial centroids
that are close together or outlying points. Random Partition initialization
varies less between runs, but this is an obvious byproduct of the tendency to
position initial centroids towards the overall center of the data set. This also
means successive runs of Random Partition tend to produce similar
initializations. If such central initializations cause Lloyd's algorithm to
converge to a local minimum with poor within cluster variance, then repeated
runs using Random Partition are unlikely to produce any noticeable improvement.
The idiosyncrasies of a particular data sets may make it appear as if Random
Partition tends to produce lower within cluster variance than any one Forgy
initialization [^7], but such a conclusion fails to account for the similar,
centrally located initial centroids of each Random Partition initialization.
You might just get lucky and Random Partition initialization converges to a
good clustering. The same might occur with a Forgy initialization, but with
more variability. Both methods are still generally bad.

Since Lloyd's algorithm is highly sensitive to initial conditions, as discussed
below, careful initialization methods can greatly improve results. More
principled approaches that address, albeit heuristically, the issue of finding
initial centroids more representative of the cluster structure are discussed
below.

### Properties
{: .subsubsection}

For reference, here are some important properties of Lloyd's algorithm:

 - *Hard Assignment* &mdash; although the k-means problem specifies that the
 data set should be partitioned into disjoint subsets, the assignment step in
 each iteration of Lloyd's algorithm does not relax this requirement. Each
 iteration's hard assignment influences subsequent update steps, and thus the
 possible clustering solutions found.
 - *Batch Processing* &mdash; all points are assigned to clusters before any
 centroid update occurs. As a consequence, the result of the algorithm is
 independent of the order in which the data are stored; for fixed initial
 centroids, permuting the index labels {{ils}}\{1, \dots, n\}{{ile}} has no
 effect on the solution found.
 - *Deterministic After Initialization* &mdash; while the initialization method
 used may be stochastic, once initial centroids have been generated Lloyd's
 algorithm is entirely deterministic. Thus, the initial centroids completely
 determine the final clustering.
 - *Complexity* &mdash; time complexity of {{ils}}\Theta\left(nkdt\right){{ile}}
 and space complexity of {{ils}}\Theta\left(d(n + k)\right){{ile}},
 where {{ils}}n{{ile}} is the number of data points, {{ils}}k{{ile}} is the
 number of clusters, {{ils}}d{{ile}} is the number of features,
 and {{ils}}t{{ile}} is the number of iterations.

On data sets that have high measures of clusterability [^10][^11], Lloyd's
algorithm converges in an acceptably small number of iterations {{ils}}t{{ile}}.
Empirical results suggest that often convergence actually does occur in
relatively few iterations. Even for dimensions as low as {{ils}}d = 2{{ile}},
however, data sets can be constructed that demonstrate in the worst
case {{ils}}t = 2^{\Omega(n)}{{ile}} [^12]. The apparent discrepancy between
these empirical and theoretical results is reconciled by a smoothed analysis of
Lloyd's algorithm [^13].

Worst case analysis can be framed as an adversarial game. In this setting,
you are player 1 and will use Lloyd's algorithm to perform k-means clustering
on some data set; player 2 is some jerk who knows all this and tries to
generate a data set that will require you to perform as many iterations as
possible. A geometric construction in the plane can be used by such a jerk to
cause you to take at least exponentially many
iterations {{ils}}t = 2^{\Omega(n)}{{ile}} [^12]. This planar construction can
be extended to higher dimensions, so you're not wining this game by using bigger
pieces.

Smoothed analysis is a hybrid of worst-case and average-case analysis.
Worst-case complexity only focuses on the worst possible input, regardless of
how rare such an input may be. Average-case complexity, on the other hand,
assumes some probability distribution over all possible inputs, and then
considers the expected outcome. Typically, a uniform distribution over inputs
is assumed &mdash; I suppose this can be seen from a Bayesian viewpoint as an
'uninformative' prior, but that just raises the reference class problem.
Smoothed complexity avoids the pessimism of worst-case complexity and the
arbitrariness of average-case complexity, while retaining positive aspects of
both. If worst-case complexity asks, 'what is the longest possible running time?',
and average-case complexity asks, 'what is the expected running time?', then
smoothed complexity asks, 'what is the longest expected running time, given that
your adversary can choose any input which will then be perturbed?'[^14]

So for smoothed complexity, the game changes: there's still some jerk trying to
generate the worst possible data for you to cluster, but now the jerk's input
gets corrupted by Gaussian noise before it gets to you. More concretely, every
data point {{ils}}x_{i} \in \mathbb{R}^{d}{{ile}} gets independently perturbed
by a {{ils}}d{{ile}}-dimensional Gaussian random
vector {{ils}}\mathcal{N}(\mathbf{0}, \sigma^{2}\textbf{I}_{d}){{ile}} before
you see it. Using smoothed analysis on Lloyd's algorithm, it turns out that the
expected number of iterations is[^13]:
<div>
$$
E[t] = O(\frac{n^{34}\log^{4}(n)k^{34}d^{8}}{\sigma^{6}}).
$$
</div>
So, Lloyd's algorithm has a smoothed running time that is polynomial in the size
of the data set, which explains the typically fast convergence in practice. Most
implementations in popular packages in languages such as R, python, and MATLAB
also typically set a maximum number of iterations, so just in case a jerk did
generate your data, you don't have to play. Alternatively, the smoothed analysis
of Lloyd's algorithm shows that if we add some Gaussian noise to our data set,
we might reduce the number of iterations to convergence [^12][^13][^15].

The determinism of Lloyd's algorithm combined with hard assignment and
batch processing cause any clustering results obtained to be highly sensitive to
initial conditions. The next section addresses these details.

### Solution Characteristics
{: .subsubsection}

When describing the clusters found by Lloyd's algorithm, three traits tend to
come up repeatedly: locally optimal, spherical, and equally sized.

Lloyd's algorithm guarantees only locally, and not globally, optimal solutions.
Although a nearly universal description of the clusters returned by Lloyd's
algorithm, the use of the analytic notion of local optimality may seen
incongruous for a combinatorial problem. We can make this statement about local
optimality rigorous by thinking of the problem in the right context.

Recall that the objective from the k-means problem is to find the lowest total
within cluster variance:
<div>
$$
\sum_{j = 1}^{k}\sum_{x \in P_{j}}\vert\vert x - \bar{x}_{j}\vert\vert_{2}^{2}. \tag{3}\label{3}
$$
</div>
Our given data set {{ils}}\{x_{i}\}_{\scriptsize{i=1}}^{\scriptsize{n}}{{ile}}
has a finite number of points {{ils}}n{{ile}}. From our earlier discussion about
Stirling numbers \eqref{2}, although the number of partitions may be huge,
it's still finite. Now as Lloyd's algorithm only uses points in the data set
to estimate each cluster mean, the set of possible centroids is also obviously
finite. Since Lloyd's algorithm can only change any estimate of a centroid
mean {{ils}}\bar{x}_{j}{{ile}} by changing which points are assigned to its
cluster {{ils}}P_{j}{{ile}}, it can only make discrete, not continuous, changes
to those estimates &mdash; a defining property of combinatorial optimization
problems.

Local optimality describes solutions that are better than any other 'nearby'
solutions, but in order for this idea to make sense we need a well-defined
concept of 'nearness'; specifically, we need to be able to talk about
arbitrarily 'near' solutions, which is equivalent to being able to move
continuously from one solution to another in arbitrarily small steps. I'm
talking calculus, and we've entered the realm of analysis. As we just saw above,
however, the combinatorial nature of the k-means problem only permits discrete
jumps in the value of our objective function \eqref{3}. So what does everyone
mean when they use local optimality to describe the output of Lloyd's algorithm?

Well, in order to make sense we're going to have to get weird. Lloyd's algorithm
estimates a set of
centroids {{ils}}\{\bar{x}_{j}\}_{\scriptsize{j=1}}^{\scriptsize{k}}{{ile}}. To
discuss the local optimality of this solution, we're going to embed it in a
vector space.

If we look inside the solution Lloyd's algorithm gives us, it consists of a set
of k different {{ils}}d{{ile}}-dimensional vectors {{ils}}\bar{x}_{j}{{ile}}. To
perform the desired embedding, we're going to vectorize this solution set:
concatenate all k centroid vectors into one long {{ils}}dk{{ile}}-dimensional
vector

<div>
$$
\bar{\xi} = (\bar{x}_{11}, \dots,\, \bar{x}_{1d},\, \bar{x}_{21}, \dots,\, \bar{x}_{2d}, \dots,\, \bar{x}_{k1}, \dots,\, \bar{x}_{kd})'.
$$
</div>
{{begin_aside}}
Technically, the {{ils}}\bar{\xi}{{ile}} need to be defined as an equivalence
class of vectors in {{ils}}\mathbb{R}^{dk}{{ile}}. Because the order of the labels of our
centroids {{ils}}\{1, \dots, k\}{{ile}} is arbitrary, we could permute the {{ils}}k{{ile}} centroids
before we concatenate them to form a different point {{ils}}\bar{\xi}' \in \mathbb{R}^{dk}{{ile}}, but
this {{ils}}\bar{\xi}'{{ile}} still corresponds to the same partition.
{{end_aside}}
Now all of the possible partitions of our data set corresponds to a discrete set
of possible points {{ils}}\bar{\xi}\in\mathbb{R}^{df}{{ile}}. Starting from any
intialization, it should be easy to see that after each assignment step, the
cost \eqref{3} is either reduced or constant. Similarly, the
variance minimizing property of means implies that after each update step, cost
is either reduced or constant. So overall, each iteration
through the main loop of Lloyd's algorithm cannot increase the value of the
objective; since we only have a finite number of possible objective values, we
will eventually reach an objective value no lower than the previous value, which
will cause Lloyd's algorithm to terminate.

So every iteration changes {{ils}}\bar{\xi}{{ile}} and lowers the cost until it
can't be lowered anymore &mdash; but the final {{ils}}\bar{\xi}{{ile}} returned
may not yield the lowest possible cost \eqref{1} for the given data set. What's
going on?

Well, this is where the hard assignment, batch processing, and determinism of
Lloyd's algorithm come in. Initialization picks
some {{ils}}\bar{\xi}^{(0)}{{ile}} from the discrete set of all
possible {{ils}}\bar{\xi}{{ile}}. Now at each successive
iteration {{ils}}t{{ile}}, the current cost defines a neighborhood of
accessible {{ils}}\bar{\xi}{{ile}} from which Lloyd's algorithm can choose; if
one of these {{ils}}\bar{\xi}{{ile}} reduces the objective, the assignment step
essentially computes a new value for \eqref{3} and the update step moves our
long centroid vector
from {{ils}}\bar{\xi}^{(t)}{{ile}} to {{ils}}\bar{\xi}^{(t+1)}{{ile}}. Because
of the structure of Lloyd's algorithm, every {{ils}}\bar{\xi}^{(t+1)}{{ile}} is
completely determined by {{ils}}\bar{\xi}^{(t)}{{ile}}.

What happens is that eventually, Lloyd's algorithm reaches a
point {{ils}}\bar{\xi}^{(t')}{{ile}} whose neighborhood contains no other {{ils}}\bar{\xi}{{ile}}.
In the language of analysis, we have reached an open neighborhood
of {{ils}}\bar{\xi}^{(t')}{{ile}} where the cost is constant. In this sense, the
centroids returned by Lloyd's algorithm corresponding
to {{ils}}\bar{\xi}^{(t')}{{ile}} yield a local minimum of the cost function.

The above argument moves the combinatorial optimization procedure used
by Lloyd's algorithm into the context of gradient descent [^16]. This gradient
descent formulation also makes the size of the neighborhoods in each iteration
more concrete. It turns out the learning rate in the {{ils}}d{{ile}}-dimensional
subspace of {{ils}}\mathbb{R}^{dk}{{ile}} corresponding to {{ils}}\bar{x}_{j}{{ile}}
is equal to {{ils}}\frac{1}{\vert P_{j} \vert}{{ile}}. In other words, the step
size in the part of {{ils}}\bar{\xi}{{ile}} that
represents {{ils}}\bar{x}_{j}{{ile}} is defined by the number of points
currently assigned to cluster {{ils}}P_{j}{{ile}}, and all k step sizes define
the neighborhood of {{ils}}\bar{\xi}{{ile}} described above[^16].

Another aspect of the clusters returned by Lloyd's algorithm is their tendency
toward spherical shape. The shape of the clusters is a direct consequence of the
k-means objective and the hard assignment used.

For a fixed point {{ils}}\hat{\mu}_{j}{{ile}}, an elementary property of the
Euclidean metric is that all points a fixed distance {{ils}}r{{ile}}
from {{ils}}\hat{\mu}_{j}{{ile}} form a sphere centered at {{ils}}\hat{\mu}_{j}{{ile}}
with radius {{ils}}r{{ile}}. So in the assignment step, for each
point {{ils}}x_{i}{{ile}}, Lloyd's algorithm essentially forms a sphere centered
at each {{ils}}\hat{\mu}_{j}{{ile}} with a radius
of {{ils}}\vert\vert x_{i} - \hat{\mu}_{j}\vert\vert_{2}{{ile}} and chooses the
sphere with the smallest radius to determine assignment.

Obviously, the specific location of the points in the data set will determine
the actual shape of the clusters found by Lloyd's algorithm. But for each
cluster, if we take the point in that cluster farthest from its centroid, we can
form a sphere containing all points in the cluster.

The last common attribute of clusters resulting from Lloyd's algorithm we will
consider is the claim that they tend to be of equal size. This description tends
to cause confusion, as the notion of size intended is often left undefined. The
centroids produced by Lloyd's algorithm define a centroidal Voronoi tessellation:
we can split up {{ils}}\mathbb{R}^{d}{{ile}} into k cells, where
the {{ils}}j^{th}{{ile}} cell consists of all the points closer
to {{ils}}\hat{\mu}_{j}{{ile}} than any other centroid. When describing the
'size' of clusters, visualizations where {{ils}}d=2{{ile}} are often used, with
each cell distinctly colored. When plotted just right, the area covered by each
color can be made roughly equal, and the equal size claim is considered
justified. Of course, such a justification is absurd: some of the voronoi cells
are unbounded and so have infinite area. All you have to do is move around the
plot, and the proportion of each color will change.

We can attempt to correct this error by drawing a box around our data set with
lengths defined by the smallest and largest coordinates in each dimension.
While this forces each Voronoi cell to be compact, it doesn't capture the real
notion of 'size' used by Lloyd's algorithm; besides, the area of those
formerly infinite Voronoi cells are now governed by the outliers in our data.

Some descriptions instead use the number of points in each cluster to define
their sizes, but this too does not completely describe the results of Lloyd's
algorithm: it is easy to construct examples of clusters containing very
different numbers of points, but which produce lower total within cluster
variance than a more equal division of point into clusters.

Combining the variance and the cardinality of the clusters in a certain way,
however, does give us a strict meaning of 'size' which can describe the behavior
of Lloyd's algorithm. For a given data
set {{ils}}\{x_{i}\}_{\scriptsize{i=1}}^{\scriptsize{n}}{{ile}}, consider
the {{ils}}j^{th}{{ile}} cluster {{ils}}P_{j}{{ile}}. As before, the
unnormalized variance within this cluster is given by its total sum of square
errors, {{ils}}\scriptsize{\displaystyle\sum_{x \in P_{j}}}\normalsize\vert\vert x - \bar{x}_{j}\vert\vert_{2}^{2}{{ile}}.
What happens if we add some point {{ils}}x^{\ast}{{ile}} to cluster {{ils}}P_{j}{{ile}}?

Well, if you work out some mildly tedious algebra, you'll find that after adding
point {{ils}}x^{\ast}{{ile}} to cluster {{ils}}P_{j}{{ile}}, its total sum of
square errors becomes [^17]
<div>
$$
\scriptsize{\displaystyle\sum_{x \in P_{j}}}\normalsize\vert\vert x - \bar{x}_{j}\vert\vert_{2}^{2} +
\scriptsize{\frac{\vert P_{j}\vert}{\vert P_{j}\vert + 1}}\normalsize{\vert\vert x^{\ast} - \bar{x}_{j}\vert\vert_{2}^{2}}. \tag{4}\label{4}
$$
</div>
That scale factor of {{ils}}\hspace{0.2em}^{\vert P_{j}\vert}/_{\vert P_{j}\vert + 1}{{ile}} actually
accounts for the change in the cluster mean caused by the presence of {{ils}}x^{\ast}{{ile}};
that tedious algebra I mentioned shows that replacing the new
mean {{ils}}\bar{x}_{j}'{{ile}} with the old mean {{ils}}\bar{x}_{j}{{ile}} is
equivalent to multiplying the square error of the new
point {{ils}}x^{\ast}{{ile}} by that scale factor, i.e. \eqref{4} is equal to
<div>
$$
\scriptsize{\displaystyle\sum_{x \in P_{j}}}\normalsize\vert\vert x - \bar{x}_{j}'\vert\vert_{2}^{2} +
\vert\vert x^{\ast} - \bar{x}_{j}'\vert\vert_{2}^{2}. \tag{5}\label{5}
$$
</div>

Using this expression for the change in total sum of square errors, we can see
that the 'size' meant when describing the clusters returned by Lloyd's algorithm
is a product of both the number of points in each
cluster {{ils}}\vert P_{j}\vert{{ile}} and its unnormalized variance. The batch
processing of Lloyd's algorithm means that two iterations are required for this
information to be used by the algorithm. When a new point is assigned to some
cluster {{ils}}\vert P_{j}\vert{{ile}}, the update step has to re-estimate the
mean before the change to the cluster's total sum of squares is reflected, and
only then is this information usable by the next iteration's assignment step.

To make things more concrete, let's compare two
clusters {{ils}}P_{j_{1}}{{ile}} and {{ils}}P_{j_{2}}{{ile}} in the special case
where we only need to assign one point {{ils}}x^{\ast}{{ile}} to one of these
clusters[^18]. Assume both clusters have the same total sum of square errors, and the
distance from {{ils}}x^{\ast}{{ile}} to each cluster's centroid is equal:
<div>
$$
\begin{align}
\scriptsize{\displaystyle\sum_{x \in P_{j_{1}}}}\normalsize\vert\vert x - \bar{x}_{j_{1}}\vert\vert_{2}^{2} &=
\scriptsize{\displaystyle\sum_{x \in P_{j_{2}}}}\normalsize\vert\vert x - \bar{x}_{j_{2}}\vert\vert_{2}^{2},\\[0.2em]
\vert\vert x^{\ast} - \bar{x}_{j_{1}}\vert\vert_{2}^{2} &= \vert\vert x^{\ast} - \bar{x}_{j_{2}}\vert\vert_{2}^{2}.
\end{align}
$$
</div>
In this case, the cardinalities {{ils}}\vert P_{j_{1}}\vert{{ile}} and {{ils}}\vert P_{j_{2}}\vert{{ile}} of
each cluster become the determining factor. Since we're only assigning {{ils}}x^{\ast}{{ile}} and
the distances to each centroid are equal, Lloyd's algorithm will break the tie
and assign {{ils}}x^{\ast}{{ile}} to one of the clusters arbitrarily. Let's
say {{ils}}x^{\ast}{{ile}} was assigned to {{ils}}P_{j_{1}}{{ile}}. Following the
update step, the sum of square errors for cluster {{ils}}P_{j_{1}}{{ile}} has increased
by {{ils}}\scriptsize{\frac{\vert P_{j_{1}}\vert}{\vert P_{j_{1}}\vert + 1}}\normalsize{\vert\vert x^{\ast} - \bar{x}_{j_{1}}\vert\vert_{2}^{2}}{{ile}}.

If {{ils}}\vert P_{j_{1}}\vert > \vert P_{j_{2}}\vert{{ile}},
then {{ils}}\frac{\vert P_{j_{1}}\vert}{\vert P_{j_{1}}\vert + 1}\vert\vert x^{\ast} - \bar{x}_{j_{1}}\vert\vert_{2}^{2} >  \frac{\vert P_{j_{2}}\vert}{\vert P_{j_{2}}\vert + 1}\vert\vert x^{\ast} - \bar{x}_{j_{2}}\vert\vert_{2}^{2}{{ile}}, and
the total sum of square errors can be reduced by moving {{ils}}x^{\ast}{{ile}} to {{ils}}P_{j_{2}}{{ile}}. So
if the sums of squares of our two clusters are equal, {{ils}}x^{\ast}{{ile}} is
assigned to the cluster with fewer points in at most two iterations.

Of course, if the sums of square errors of our two clusters are not equal, then
the behavior of Lloyd's algorithm is a more complicated interplay between the
square errors and cardinalities of the different clusters. The spherical shape
of the clusters can provide some insight into this aspect of a cluster's size.
And of course, assigning only one point to one of two clusters is not a typical
iteration of LLoyd's algorithm. But the above ain't just shootin' the breeze
&mdash; that would be a waste of ammunition. It illustrates that the size of a
cluster is determined by both the spread of the points from its centroid, as
quantified by the sum of square errors, and the number of points in the cluster.
In trying to minimize the total sum of square errors, Lloyd's algorithm will
tend to produce clusters of roughly equal size in precisely this sense.

### The Measure of a Mistake
{:subsubsection}

We have so far covered some of the most salient properties of the k-means
problem and the method most commonly associated with it, Lloyd's algorithm.
Hopefully, you now have a firm grasp on both, as well as the distinction between
them. This is a natural point at which to address a mistake that appears fairly
common in both the literature and implementations, and which ultimately
originates from an ignorance of the topics we've just covered.

If you look back at equation \eqref{1}, you'll see that the k-means problem is
one of variance minimization. The objective is to find a partition of our data
into k clusters that produces the lowest possible total within cluster variance.
Since the means of each cluster are the unique minimizers of the variance, they
naturally show up.

The number of candidate partitions is huge, {{ils}}O(k^n){{ile}} from bounds on
Stirling's numbers of the second kind. Using those equivalence class vectors of
means {{ils}}\bar{\xi} \in \mathbb{R}^{dk}{{ile}} from our discussion of local
optimality, a combinatorial argument can be applied to the algebraic surfaces
defining the boundaries of the Voronoi tesselations corresponding to each
partition, and our bound can be improved to {{ils}}O(n^{dk + 1}){{ile}} [^19].
That's still huge. So we resorted to heuristic methods &mdash; sacrificing
optimality for speed. The heuristic we've studied is Lloyd's algorithm. This
procedure makes some further assumptions not included in the k-means problem,
some pertinent consequences of which we've examined above.

Examining the assignment step, Lloyd's algorithm can be classified as a greedy
optimization method: each {{ils}}x_{i}{{ile}} minimizes its own square error
with gluttonous disregard for the square errors of any other
point {{ils}}x_{\ell}{{ile}}, where {{ils}}\small\ell \neq i{{ile}}.

Since every point behaves greedily during the assignment step, some simple
algebra shows that minimizing the square error to a centroid is the same thing
as minimizing the *Euclidean* distance. Just take the square root, bada bing
bada boom, you've got the distance. Since the square root is monotonic, the
minimizer of one is the minimizer of the other:
<div>
$$
\argmin_{\scriptsize\mu \in \{\mu_{i}\}_{1}^{k}}\vert\vert x_{i} - \mu \vert\vert_{2}^{2} = \argmin_{\scriptsize\mu \in \{\mu_{i}\}_{1}^{k}}\vert\vert x_{i} - \mu \vert\vert_{2}
$$
</div>

And this is where the trouble starts. That the assignment step selects the
nearest centroid for every point is incidental, a tautology of algebra. When
explaining what Lloyd's algorithm does, more internet posts than I'd like to
remember describe the assignment step *only* as finding the nearest centroid.
Just check out any Medium article on k-means. Sure, this is an equivalent way of
describing the assignment step, but it is actually a consequence of the specific
heuristics Lloyd's algorithm uses: splitting the estimation task into separate
assignment and update steps, and using greedy optimization in assignment. This
heuristic works fairly well in minimizing the true k-means objective, the total
within cluster variance, but as we've already seen it does not guarantee finding
the true minimum. Unfortunately, many people conflate the algorithm with the
problem, and relegate themselves to a superficial understanding polluted by
epiphenomena.

The first issue that arises from this incomprehension commonly shows up when
someone tries to roll their own version of Lloyd's algorithm &mdash; they'll
actually perform a square root operation for every point and centroid pair in
the assignment step to find the distance. These are garbage operations. Even if
your square root implementation conforms to IEEE 754 or 854 floating point
standards with exact rounding [^20], you're still introducing an unnecessary
operation and losing precision. Those square roots might be inconsequential on
toy data sets, but for massive data sets and distributed processing this is
more likely to be an issue.

Now if you take that first mistake, pretend it's insight, and smugly elaborate on
it, you'll get the second, and much worse, error stemming from this confusion:
swapping arbitrary non-Euclidean metrics in place of the square error in the
assignment step while leaving the update step unchanged. If the last mistake was
garbage, this one is the dump that garbage took. Both the assignment and update
steps of Lloyd's algorithm try to minimize variance &mdash; they each have the
same objective in mind. If instead of the square error, the assignment step
uses, say, the {{ils}}L_{1}{{ile}} metric to determine the 'nearest' centroid
for every point, the assignment and update steps of Lloyd's algorithm now have
two different objectives. You know what minimizes the {{ils}}L_{1}{{ile}} error?
The median. Since the result of one step provides the parameters of the other
step, this just creates a cycle of compounding bad estimates. That guarantee
that we'd at least find a locally optimal solution using Lloyd's algorithm?
Gone. You're bada bing bada screwed.

Unfortunately, this perversion of Lloyd's algorithm seems rampant &mdash; again,
just check any Medium article on k-means. Such ridiculous methods also show up
in the literature [^21][^22][^23][^24], as well as in primary implementations in
MATLAB and mathematica. The MATLAB *documentation* contains an example of an
'analysis' using the Manhattan distance. MATLABS kmeans algorithm tries to
correct for the potential bunkum of swapping in a non-Euclidean metric by
performing a 'corrective' pass over every data point at the end. Good luck with
that. The amap package in R, which according to rdocumentation.org is in the
95th percentile in terms of overall downloads, has a kmeans implementation that
allows arbitrary distance functions with an inefficient implementation of
Lloyd's algorithm as a bonus. The kmeans function in amap should never be used;
the standard stats package's kmeans function is a far better implementation,
which also uses the Hartigan-Wong algorithm by default.

Listen, sure, you'll get some results. Hell, if you cap the iteration number,
by definition your algorithm will 'converge.' But like my grandmammy said,
just cause you put a cat in the oven, don't make it a biscuit. The convergence
of Lloyd's algorithm for the k-means problem is based on both steps operating on
the same objective - minimizing cluster variance. If you use a non-Euclidean
metric in the assignment step, then this alteration has to be reflected in an
appropriate modification of your update step. Now, there are in fact modified
versions of Lloyd's algorithm that do just that. Unfortunately, the term
'k-means' is sometimes used to describe these procedures, even though the
cluster centers are no longer given by the arithmetic means of the cluster
points. But at least these procedures make sense. The mathematical soundness of
some such generalizations can be grounded in the theory of Bregman divergences,
which I'll touch on when discussing the connection of k-means algorithms to
Gaussian mixture models.

## Sequential K-Means
{: .subsection}

As mentioned above, Lloyd's algorithm is a batch algorithm &mdash; only after
every data point has been assigned to a cluster do we update our estimates of
the centroids. This can lead to large changes in the centroid estimates, and can
cause large memory overheads. An alternative approach is to interleave the
assignment and update steps: after assigning a point to a cluster, update the
centroid estimate immediately.

### Properties
{: .subsubsection}

### Solution Characteristics
{: .subsubsection}

## Hartigan-Wong Algorithm
{: .subsection}

Neither the classic Lloyd-Forgy algorithm nor the online MacQueen algorithm takes
into account the potential degradation of the within cluster variance when a
single point is added to a cluster. Specifically, whenever a point is assigned
to a cluster, the estimate of the mean will change &mdash; and this will cause
the cluster variance to change not just because of the added point, but because
the square error for *all the other cluster points* will also change. The
Hartigan-Wong algorithm takes this into account and provides a better
approximation method to minimizing the total within cluster variance.

### Properties
{: .subsubsection}

### Solution Characteristics
{: .subsubsection}

## Initialization Methods
{: .subsection}

## Data Normalization
{: .subsection}

# Clustering Results
{: .section}

## Benchmarks
{: .subsection}

## Cluster Validity
{: .subsection}

# Relationship to Other Methods
{: .section}

## Gaussian Mixture Models
{: .subsection}

## Principal Component Analysis
{: .subsection}

## Expectation Maximization
{: .subsection}

# Considerations in Application
{: .section}

## Estimating k
{: .subsection}

## Alternative Methods
{: .subsection}

# Summary
{: .section}

[^1]:
    {% include citation.html key="truffle" %}
[^2]:
    {% include citation.html key="ibneck" %}
[^3]:
    {% include citation.html key="stirling2basics" %}
[^4]:
    {% include citation.html key="stirling2bounds" %}
[^5]:
    {% include citation.html key="kmeanshistory" %}
[^6]:
    {% include citation.html key="lloydpcm" %}
[^7]:
    {% include citation.html key="fourkmeansinitmeths" %}
[^8]:
    {% include citation.html key="caa" %}
[^9]:
    {% include citation.html key="compstudykmeansinitmeths" %}
[^10]:
    {% include citation.html key="effectivenesslloydtype"%}
[^11]:
    {% include citation.html key="clusterability"%}
[^12]:
    {% include citation.html key="kmeansplaneexp"%}
[^13]:
    {% include citation.html key="kmeanspolysmooth"%}
[^14]:
    {% include citation.html key="smoothedml" %}
[^15]:
    {% include citation.html key="howslowkmeans" %}
[^16]:
    {% include citation.html key="kmeansconvergenceprops" %}
[^17]:
    {% include citation.html key="optmethsclustanal"%}
[^18]:
    {% include citation.html key="methclustnums"%}
[^19]:
    {% include citation.html key="weightedvoronoi" %}
[^20]:
    {% include citation.html key="everycsfloatpoint" %}
[^21]:
    {% include citation.html key="kmeans3dists"%}
[^22]:
    {% include citation.html key="kmeansdiffdists"%}
[^23]:
    {% include citation.html key="kmeansbadtechnique"%}
[^24]:
    {% include citation.html key="kmeansperfdist"%}
<script src="/assets/js/d3.js"></script>
<script src="/assets/js/elements/Unsupervised/Cluster_Analysis/kmeans.js"></script>
