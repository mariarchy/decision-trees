# Welcome!

## If you're reading this

This is my attempt at building my mental model of a supervised learning algorithm called Extremely Randomized Forests (commonly referred to as ExtraTrees) after absolutely bonking a coding assessment for a fellowship program I had my eyes set on (I'm fine, I swear).

## The exercise

I've spent the last few hours reading articles on the topic ([this visualization](https://mlu-explain.github.io/decision-tree/) has to be my favourite thus far). My goal is to implement the algorithm from scratch and stress-test my understanding of the concept. 

Here are a few rules I set for myself:
- Do not rely on the use of LLMs to write code or tests. In other words, build the algorithm from scratch. I want to reveal the gaps in my knowledge as soon as they appear. I rely on readings and Google search to resolve any gaps in understanding I've identified.
- Do not look at coding examples online.

My goal is to approach this exercise genuinely and honestly. I do not want to appear as anything other than myself — and what that entails is not posturing more than I actually know.

## Disclaimer

I am a senior software engineer with little-to-no ML experience. I don't expect anything to come of this exercise other than my own enrichment. I took the assessment as meaningful signal that I have much to learn and I'm here to genuinely engage with what I don't know.

## Decision trees, in my own words

Machine learning, or at least my mental model of it thus far, is interesting. You're postulating that the underlying pattern to be learned exists in your training data, without knowing the full specification of that pattern. These various methods of learning attempt to learn something neither the algorithm nor the human can see, by analyzing a sample of its shadows.

Decision trees are a class of machine learning methods that take a stab at this problem of learning through a deicison tree structure. The underyling principle across these approaches is that the _truth_ or the _pattern_ you're trying to uncover can be modeled as a tree of decisions.

The classic example is a decision tree for classifying fruits. Imagine you have a table of data on fruits and their weights:

```
        weight (g)   diameter (cm)    label
ex 1:    150           7              apple
ex 2:    160           7              apple
ex 3:    140           6              apple
ex 4:    180           8              apple
ex 5:    200           9              orange
ex 6:    220           10             orange
ex 7:    210           9              orange
ex 8:    230           10             orange
```

This is a reasonable decision tree you can construct from the data:

```
   weight < 190?
    /         \
  yes          no
   |           |
 apple       orange
```

In the tree above, we have one decision node: "weight < 190". This node contains a target feature: "weight" and a split "190". The left subtree (in this case, the "apple" leaf node) contains the set of examples for which the condition (target feature < split) holds true. The right subtree contains all other examples in the set (for which the condition is false).

But what if you encounter an apple with a weight of 250g? If you want 100% training accuracy, you'd have to update your decision tree to look like this.

```
   weight < 190?
    /         \
  yes          no
   |           |
 apple     weight < 250
            /         \
          orange     apple
```

We've effectively hardcoded or overfit our decision tree to the outlier. "weight < 250" is not necessarily a meaningful rule about the underlying _truth_, i.e. fruits > 250g are likely apples. We've simply created a new rule to accommodate this outlier in our dataset. And this is one of the gotchas of decision tree algorithms: you risk overfitting your decision tree to the data and uncovering hardcoded, one-off rules about the truth, ones that don't generalize to the real world, i.e. your test data. These trees are typically many layers deep.

### And so, what are the classes of improvements we can make?

1. **Recognize the whole is better than the sum of its parts.** The probability of multiple bad trees predicting the wrong result is lower than that of an individual bad tree. Construct a "forest" of such trees and let them vote on the correct answer. The majority vote is likely correct. These methods are called "ensemble" decision trees.

2. **Improve the construction of a single tree**, e.g. choose nodes (features and splits) that lead to the most information gain (measured by entropy loss in the left and right split sets).

The goal of the approaches within #1 must optimize for (1) the diversity and (2) independence of the trees. For instance, if your trees are all bad and strongly correlated, the probability of the majority vote being correct doesn't change much from the probability of a single tree being correct. Diversity and independence is achievable through randomizing as much of the tree construction as possible.
 
## Extremely Randomized Forests, in my own words
Extremely Randomized Forests is an algorithm under the set of approaches within #1. In layman's terms, it says, "Let's create a bunch of trees randomly, i.e. randomize the feature and the split we choose at each internal node — capped at the minimum size a subtree of samples is allowed to contain — and let this forest of random trees vote on prediction. Majority vote wins"

### Algorithm

First, we need to build each tree. The algorithm goes as follows:
1. Start with the root node. Each node has a subset of the training dataset it works upon. At the root, this is the full training dataset. 
2. Choose a random subset of features for this node (typically the square root of the total features available to the subtree). 
3. For each feature in the subset, choose a random split value. The range should be the min to max value of feature values in the sample for this node.
4. Pick the best among the random splits by measuring the impurity reduction, i.e. the split that gets us closest to a unanimous result, i.e. all examples in the left tree are one class, all examples in the right tree are another class.
5. Create a left and a right subtree using the best split.
6. If any of the following conditions are met in the subtree, create a leaf node. No more subtrees allowed!
- The subtree has hit maximum purity, i.e. the results are unanimous for the subtree.
- No subtrees yield a reduction in impurity, i.e. it's not worth going further.
- We have hit the maximum allowed depth for the tree.
- We have hit the maximum permitted leaf nodes in the tree.
7. Otherwise, repeat step 1-4 for each subtree with their selected subset of samples.

Build any number of such trees.

To generate the prediction yielded by a single tree, trace your tree through your example until you hit a leaf node. Return the leaf node and its associated class. To generate a prediction yielded by the forest of trees, repeat the single-prediction process on all trees and return the majority vote.

### Debugging log

Hello! Reporting live from the ground. There are a few special cases to consider.

1. When sampling features in step 2, sample from _constant_ features. Otherwise, we will not find a valid split value for our subtree.
2. If all feaures are constant, create a leaf node. There's no further split required.
3. We should ignore splits with a score of 0, i.e. splits that do not improve the purity of the subtree

### Scope

For the purposes of this exercise, I will not enforce maximum leaf nodes.

## Data models

OK so to implement this, we need a few data models:

- `InternalNode`, the internal node with a split feature, value, and a left and right subtree
- `LeafNode`, the leaf node with an assigned class
- `RandomizedTree`, a single randomized tree 


### InternalNode

The internal node contains:
- A pointer to the left and right node, which can be an `InternalNode` or `LeafNode`
- A split feature
- A split value

### LeafNode

The leaf node contains just a class label.

### RandomizedTree

The randomized tree that implements the ExtraTree algorithm. We'll just store a pointer to the root.

Across these models we can store additional properties, but I'll just stick to the data I need for this exercise.