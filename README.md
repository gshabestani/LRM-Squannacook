# LRM-Squannacook

Input data for generating stochastic streamflows are observed and simulated timeseries of streamflow. their format needs to be CSV with 2 columns for observed and simulated flow named "Qgage" and "Qmodel" respectively.

The "generatingQ.py" generates stochastic streamflows with use of LRM model (for more information on the model see Shabestanipour et.al 2022 (Submitted)).
The 'monthlyKNN.py' generates stochastic streamflows with use of monthly knn LRM model.

For verifying and validating the stochastic flow generated with SWM or SWM_knn, use the file called "SWM_verify_validate.py".
