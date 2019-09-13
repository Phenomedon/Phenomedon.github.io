---
layout: post
title: 'K-means Clustering'
topic: 'Cluster Analysis'
weight: 1

asides:
  - The map is not the territory
---
You've just loaded a fresh dataset. It's still so hot from transformation that
your laptop fans haven't stopped whirring. After performing some basic visual
inspection of the data, you wonder if some samples might be related to each
other in terms of the values of their attributes. There might even be some sort
of underlying subset structure lurking in your data that would allow you to
group them into different categories according to the attribute values taken by
related samples.

In other words, it's cluster hunting season.

Clusters are like truffles. And like any good fungus hunter knows, the first
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

<div style="position: relative; width: 100%;">
 <div style="position: absolute; left: 100%; top: 0; transform:translateY(-50%)">
   <aside style="width:100%;">This aside is in the right spot?</aside>
</div>
</div>

# K-Means Problem
