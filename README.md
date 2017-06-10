### Introduction

The dataset consists of iTunes apps details such as paid/free app, store-release
date and ranking. The GlobalRank is a metric assigned daily by Adjust to these
apps, based on their store performance. For example, if an app ranks #10 on the
US iTunes store in genre Games for a given day, that rank will likely weigh more
towards the GlobalRank than another app's rank in genre Utility in Greece. This
is due to the popularity of the US app-store and the genre Games v.s. Utility
apps in Greece (the situation in 2016, at least).

The best GlobalRank is of course #1 and while there are millions of apps on the App
Stores, only a few hundert thousand actually perform well enough to be ranked at
all. In this set only the top 1000 GlobalRanks by day for 2016.

### Data description

Itunes Applications - here `release_date` is the store release date of an app.
So it can go back to 2008 for some of the first apps released on the market.
Also app name and paid/free download flag are given per app.

### Challenge

The challenge is to research the given data. Impress us with exploratory
statistics, a story you may find about the data, some informative visualisations
and why not even some prediction? You are free to use any tool you wish. Please
also provide the code along for your results so that we can reproduce your
research.