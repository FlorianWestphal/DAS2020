This folder contains the raw data collected for this study.
Each of the three subfolders contains the results for the three
respective datasets.

The contained txt files list the MAP for each image in the respective test set.
These files have been created with the evaluation tool of the 
[ICFHR 2016 Handwritten Keyword Spotting Competition](https://www.prhlt.upv.es/contests/icfhr2016-kws/evaluation.html).

The file names encode the selection strategy, the number of training images, 
and the query type as follows:
```
<strategy>_<number training imgs>_ipm_<query>.txt
```
In case of the random strategy (`rnd`) the additional number indicates the used
seed value.
