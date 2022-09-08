# Estimation-the-GR-law-s-b-and-a-values-through-MLE-and-simple-regression
## Python libraries used:
- Pandas
- Numpy
- statsmodels.formula.api
- matplotlib

## Project description

Script that reads a seismic catalogue, estimates the magnitude of completeness (Mc) through maximum curvature and finds the optimal parameters of the Gutenberg-Richter (GR) law, which is of the form:

logN = a + b * logM

where N is the number of earthquakes with a magnitude of at least M.

The script uses seismic events above Mc (which can be given manually or be estimated automatically). Parameters a and b are obtained through Maximum Likelihood Estimation (MLE) and linear regression with least-squares.

Maximum Curvature and Frequency-Magnitude Distribution (FMD) plots are generated to visually assess the results.

Sample data (sample_catalog.cat) where acquired from the Seismological Laboratory of the National and Kapodistrian University of Athens at:

http://www.geophysics.geol.uoa.gr/stations/gmapv3_db/index.php?lang=en

## Sample Outputs

![](img/b_value_0.0.png)
![](img/b_value_1.3.png)
![](img/b_value_2.2.png)
![](img/maximum_curvature_mc.png)
