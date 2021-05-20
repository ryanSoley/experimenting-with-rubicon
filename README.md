# experimenting-with-rubicon

This repo illustrates how `rubicon_ml`'s dashboard can make tracking
your model code easy!

`my_model.py` contains a simple Scikit-learn classification model
and trains it across a few parameter sets with a grid search.
These results are then logged with `rubicon_ml`'s `git`
integration enabled. As more commits are made to this repo that
change the classifier and the parameter sets, each group of
experiments logged by `my_model.py` will contain a direct reference
to the code on GitHub that produced them. These references are
used by the `rubicon_ml` dashboard to generate links to the code on
GitHub.

More info on the Palmer penguins dataset used in this example can
be found [here](https://allisonhorst.github.io/palmerpenguins/).

### setup

```
conda env create -f environment.yml
conda activate experimenting-with-rubicon
```

### run

```
python my_model/my_model.py
```
