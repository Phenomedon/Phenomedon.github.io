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
your laptop fans haven't stopped whirring. After performing some basic visual
inspection of the data, you wonder if some samples might be related to each
other in terms of the values of their attributes. There might even be some sort
of underlying subset structure lurking in your data that would allow you to
group them into different categories according to the attribute values taken by
related samples.

It's cluster hunting season.

Clusters are the truffles of your dataset, waiting to be found and finely grated
over your analysis. And like any good fungus hunter knows, the first
task is getting the right hog: one with a keen sense of smell, a lust tempered
by perspicacity, a snout for rooting but a maw you can muzzle. Ok, I may have
lost the thread of the analogy there, but the hogs we're talking about are
clustering methods. Very often, the first choice in clustering hogs is the
granddaddy of 'em all - k-means clustering.

Not only is k-means one of the oldest forms of cluster analysis, it is also one
of the most widely used, and frequently the first approach to unsupervised
learning taught. K-means' enduring popularity both in practice and pedagogy
stems from the elegance in its simplicity. In my experience, this unfortunately
also means that k-means is one of the most frequently misunderstood, poorly
explained, and outright abused methods in machine learning. Granted, my personal
experience, like anyone's, is statistically biased - no individual is an
arbitrarily random sampler - but I have seen enough incorrect implementations
and misleading characterizations to warrant address here. K-means may be elegant
in its simplicity, but that doesn't excuse simplicity in the minds of those who
use it.

In this article I aim to impart an appreciation for the nuances of k-means
cluster analysis: what k-means actually is, when it is appropriate, when it is
not, and how to tell the difference.

One of the major sources of confusion about k-means can be traced to conflating
the problem with the algorithm used to solve it.

# K-Means Problem

Let's go back to our hypothetical dataset. Hopefully your imaginary laptop's
fans have spun down by now. To make things more concrete, we'll say that we have <mjx-container>\(n\)</mjx-container> data points, each of which has <mjx-container>\(d\)</mjx-container> features, and each feature is a real
number; more succinctly, each data point <mjx-container>\(x_{i}\)</mjx-container> is a <mjx-container>\(d\)</mjx-container>-dimensional real
vector for <mjx-container>\(1 \leq i \leq n\)</mjx-container>.

Now suppose there *is* some sort of meaningful, useful subset structure lurking
in the data, just waiting to be discovered. Before we can find it, we need to
decide how we will operationalize the distinguishing features of the subsets;
in other words, what measurable characteristics will we use to differentiate
one cluster from another?

From a statistical point of view, an obvious choice to characterize
subpopulations is the average, or mean, of each subpopulation. Means are easy
to compute.
{{begin_aside}} Recall that a statistic is <dfn>sufficient</dfn> relative to an unknown
parameter of a particular statistical model if no other quantity computed from
a sample could provide any more information about the unknown parameter. The
notion of an <dfn>information bottleneck</dfn> is a related information
theoretic concept that quantifies how much information is lost about the
parameter of interest by using a particular statistic. In a certain sense, the
information bottleneck generalizes the concept of a sufficient statistic from
the perspective of lossy compression.
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

The 'k' in 'k-means' is the answer to the first question. We assume we know how
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
mean: it is the unique minimizer of the mean square error to all the sample points.
Formally, for a random sample of <mjx-container>\(n\)</mjx-container> observations <mjx-container>\( \{x_{i} \}_{i = 1}^{n}\)</mjx-container>,
the constant (vector) that uniquely minimizes the expected sum of the squared
errors to each observation is the mean <mjx-container>\(\mu\)</mjx-container>:

<div>
$$
\DeclareMathOperator*{\argmin}{arg\,min}
\begin{align}
\mu = \argmin_{c \in \mathbb{R}^{d}} E\bigg[\sum_{i = 1}^{n}\vert\vert x_{i} - c \vert\vert_{2}^{2}\bigg]
\end{align}
$$
</div>
We want to apply this property of means not to the whole data set, but to
each of our <mjx-container>\(k\)</mjx-container> clusters individually. So for the <mjx-container>\(j^{th}\)</mjx-container> cluster <mjx-container>\(P_{j}\)</mjx-container>, we
want to use just the mean <mjx-container>\(\mu_{j}\) of that cluster.

<div>
$$
\mu_{j} = \argmin_{c \in \mathbb{R}^{d}} E\bigg[\sum_{x \in P_{j}}\vert\vert x - c \vert\vert_{2}^{2}\bigg]
$$
</div>

The main thing to notice here is that now our sum only runs over points in the <mjx-container>\(j^{th}\)</mjx-container>
cluster <mjx-container>\(P_{j}\)</mjx-container> rather than all <mjx-container>\(n\)</mjx-container> points.

We now have an objective function for each cluster. Since we don't know the
true mean <mjx-container>\(\mu_{j}\)</mjx-container> for each cluster <mjx-container>\(P_{j}\)</mjx-container>, we have to estimate it using our
data. If you're worth your salt doing this, you know that the sample average is an
unbiased estimator of the population mean. So we just need to take the average
of all the points in cluster <mjx-container>\(P_{j}\)</mjx-container>, and we'll have our estimate for <mjx-container>\(\mu_{j}\)</mjx-container>.
<div>
$$
\bar{x}_{j} = \frac{1}{|P_{j}|}\sum_{x \in P_{j}}x
$$
</div>
But this isn't good enough - in order to estimate a cluster's mean using the
average of all data points belonging only to that cluster, we need to first know
which data points belong to that cluster. Of course, if we knew which points
belonged to each cluster, we wouldn't need to do any damn cluster analysis in
the first place! In order to address this dilemma, we incorporate the cluster
assignment information into our objective function: we seek the assignment of
points to clusters that minimizes the total *within cluster* sum of squares
error over all cluster assignments <mjx-container>\(\mathcal{P}\)</mjx-container>.
That is, out of all partitions <mjx-container>\(\mathcal{P}\)</mjx-container> of our data into <mjx-container>\(k\)</mjx-container> non-empty subsets <mjx-container>\(\mathbf{P} = \{P_{1}, P_{2}, \dots P_{k}\}\)</mjx-container>, we want to find a particular
partition <mjx-container>\(\mathbf{P}_{0}\)</mjx-container> that minimizes the overall sum of squared errors,
<div>
$$P_{0} = \argmin_{\mathbf{P}\in\mathcal{P}}\sum_{j = 1}^{k}\sum_{x \in P_{j}}\vert\vert x - \bar{x}_{j}\vert\vert_{2}^{2}.$$
</div>
Sounds simple enough, right? All we have to do is try out all the different ways
to split our data into <mjx-container>\(k\)</mjx-container> clusters, and find one that gives us the smallest
total when we add up the sum of square errors within each cluster.

Well I hope you've got a lot of free time, because the number of possible
partitions is *huge*. How huge? Mathematically speaking, it makes Dwayne Johnson
look like Kevin Hart - when he was five. The Stirling numbers of the second kind
count the number of ways to partition a set
of <mjx-container>\(n\)</mjx-container> objects
into <mjx-container>\(k\)</mjx-container> non-empty subsets.
For a given <mjx-container>\(n\)</mjx-container> and <mjx-container>\(k\)</mjx-container>,
the corresponding Stirling number of the second kind,
denoted by <mjx-container>\(n \brace k\)</mjx-container>, is given by the
formula
<div>
$$
{n \brace k} = \frac{1}{k!}\sum_{i=0}^{k}(-1)^{i}\binom{k}{i}(k - i)^{n}.
$$
</div>
See those factorials and binomial coefficients? Turns out we're dealing with
**super-exponential** growth.

<script src="/assets/js/d3.js"></script>
<script src="/assets/js/elements/Unsupervised/Cluster_Analysis/kmeans.js"></script>
