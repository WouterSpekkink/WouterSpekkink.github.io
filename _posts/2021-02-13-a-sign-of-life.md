---
layout: post
title: "A sign of life"
date: 2021-02-13 21:00 +0000
categories: Software 
tags: Q-SoPrA Non-Technical
---

## I was 'gone' for a bit
The title of this blog post is a bit overly dramatic. 
Yet, it has been almost three years since my last blog post, so it does to some extent feel to me as if I am finally giving a sign of life after a long absence. 

There are multiple reasons why I have not updated my website (except for some minor things) in a long time. 
One of them is that I became a father in 2018, which of course upended all my routines. 
Another reason is that I have changed jobs (and countries) in 2019. 
There are more reasons, but not all that interesting or important.

## But I am reviving this blog
What is more important is that I would like to pick up the routine of blogging again. 
As before, I will probably blog quite a bit about Q-SoPrA, the software that I have been working on for a while now. 
The development of Q-SoPrA is another thing that got slowed down due to the various changes in my personal life. 
However, I have done quite some work on it, and I feel like it is probably ready for a public beta release. 
However, I am not going to actually release it until after I finish a publication on the software (writing publications, another thing that got slowed down).
In the meantime there have been a few people that have been using it (including myself, of course). 
This includes a PhD student who recently graduated, as well as a master student who also recently graduated. 
One of my own projects in which I used Q-SoPrA was the evaluation of the development the Rotterdam Climate Agreement. 
We recently finished a [report][1] on this project (only available in Dutch) that includes various visualizations that I made with Q-SoPrA.
Last year I was involved in a H2020 bid in which we also included Q-SoPrA as one of our main tools.
The bid got excellent reviews, but when only 3 out of more than 50 bids can get funded, it becomes a bit of a lottery.

Anyway, I expect to be using Q-SoPrA and its underlying methodology in various future projects and I still hope that Q-SoPrA will be useful to others too.
I hope that picking up this blog again also gives a boost to my efforts to work towards a public release.

As an aside, I am also thinking about blogging on other topics related to my research, although I have to admit I find that much harder to do.
Maybe those posts will just be much shorter.

## Some things I have added to Q-SoPrA
Anyway, I wanted to keep this one short as well. 
Let me close with an overview of a few (probably not all) things that I changed in Q-SoPrA in the meantime:

* I added the possibility to distinguish between multiple coders in a project. 
This means that there are now also (still limited) possibilities to check for inter-coder reliability.
Q-SoPrA already included something like an inter-coder check for the linkages between events, but now inter-coder reliability can also be checked for [attributes][2].
* I added a visualization widget to the linkage coder (I do not believe I discussed the linkage coder in detail yet; good topic for another post) to give visual feedback on the linkages that you are coding (see below).
* A added various date validation options to the [data widget][3] that helps you to check the order of the incidents that you recorded in your dataset.
* I have added various options to layout events in the event graph widget (mentioned [here][2]) based on the date on which they occurred.
* I added various concordance plot-type visualizations. 
These can, for example, be used to visualize where in the dataset (and in time) codes occur. 
They can also be used to create quick overviews of where on a timeline events of a given type occur. 
I will probably show some of this off in a later post.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/visual_feedback.png" class="img-fluid rounded z-depth-1" zoomable=true center=true %}
    </div>
</div>
<div class="caption">
Visual feedback when creating linkages.
</div>
 
There is probably a lot of stuff that I forget now, because it has been a while since I have been able to actively work on Q-SoPrA.

There is also some bigger stuff that I am actively working on in separate branches of the code repository. 
One of these branches implements something based on truth tables that are based of event graphs. 
This is related to a paper that I am collaborating on with someone who is into set theory.
Anyway, hopefully I can write more on that later too.

Actually, while writing this I realize there is plenty of stuff that could be interesting to discuss further in future blog posts. 

## To be continued...
Okay, this was a short one, but my main purpose was to make a start again, and do it quick. Until the next one!

[1]: {{ site.url }}/assets/pdf/rka_rapport.pdf 
[2]: {{ site.baseurl }}{%post_url 2018-01-25-attributes-in-qsopra %}
[3]: {{ site.baseurl }}{%post_url 2017-12-26-getting-started-with-qsopra %}
