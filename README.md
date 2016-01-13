# PersonalityDNA
### Cluster Analysis Personality Type System
All rights to this project and all content belong to Spark Wave, LLC.

This repository is the home of a project undertaken at the behest of and with aid of [ClearerThinking.org](http://www.clearerthinking.org/). Our goal is to develop a new personality type system which has both the consistency, validity, predictive power of the Big 5 (5-factor Model) and popular appeal of personality type systems such Myers-Briggs and Enneagram. 

### Why?
People love classifying the personality type of both themselves and others, and do so a wild variety of systems including astrology/horoscopes, Myers-Briggs, Enneagram-type, Hogwarts Houses, and a myriad internet quizzes of the form "Which X are you?" The appeal is obvious. Identifying personality types appears to offer predictive and explanatory power. "Jenna and I don't get along because I'm a Cancer and she's a Pisces." "You're a type 3, which means you're always trying please people." And so on. And these attempts are not completely crazy, scientifically-researched personality systems such as the Five Factor Model (Big 5) have proven to be reliable, constistent, and predictive. (Of course not without controversy.)

It is our impression that unlike the Five Factor Model which, as the name states, is based on factors, culturally popular personality type systems such as Myers Briggs focus on archetypical personality types. We will attempt to create a new system which focuses on types rather than factors (traits), but still has predictive power and reliability.

### How?
Psychological researchers have developed extensive lists of questions which capture varying aspects of personality between people. Using factor analysis, one reduce a large number of questions and possibly traits to five. However instead of factoring responses to the personality questions, as in the FFM, one could attempt to cluster them and determine is distinct personality clusters emerge\**. This is our intention.

Initially we will analyse existing datasets of psychometric responses, and eventually we can test our work via responses on [ClearerThinking.org](www.clearerthinking.org) which has received in excess of 100,000 responses on some of its quizzes.

### Technical Details and Challenges
We are starting with a dataset mailed available by Harvard University: [Selected personality data from the SAPA-Project: 08Dec2013 to 26Jul2014](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SD7SVE)
>These data were collected to evaluate the structure of personality constructs in the temperament domain. In the context of modern personality theory, these constructs are typically construed in terms of the Big Five (Conscientiousness, Agreeableness, Neuroticism, Openness, and Extraversion) though several additional constructs were included here. Approximately 24,000 individuals were administered random subsets of 696 items from 92 public-domain personality scales using the Synthetic Aperture Personality Assessment method between December 8, 2013 and July 26, 2014. 

While the number of respondents is extremely high, this dataset has the excellent challenge of having each respondent replying to only a subset of the questions. In other words, 90% of values are missing. 

In our initial attempt to:
* Apply an appropriate form of factor analysis to the data set.
* Apply matrix factorisation techniques which are used in similar problems with recommendation systems and bioinformatics research (hence the working name Personality DNA).
* Cluster using a variety of different dissimilarity measures, starting with Euclidian distance and extending to whatever makes sense.

\** Admittedly, factor analysis and cluster analysis are closely related meaning the difference is not so long. However, the difference can framed as whereas the focus of factor analysis is to group features/traits together, the focus of clustering is to group samples/examples together.
