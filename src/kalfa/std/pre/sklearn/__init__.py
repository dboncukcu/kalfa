from kalfa.registration import pack

lego = pack(__name__)


lego("/pre/sklearn/standard_scaler", "scalers:StandardScaler", state=True, alias="standard_scaler", grouped=True,
     description="Standardize a column to zero mean and unit variance (sklearn StandardScaler); one object over every "
                 "column that names it, its statistics per column")
lego("/pre/sklearn/minmax_scaler", "scalers:MinMaxScaler", state=True, alias="minmax_scaler", grouped=True,
     description="Scale a column into [low, high] (sklearn MinMaxScaler); one object over every column that names it, "
                 "its statistics per column")
lego("/pre/sklearn/max_abs_scaler", "scalers:MaxAbsScaler", state=True, alias="max_abs_scaler", grouped=True,
     description="Scale a column by its largest absolute value, into [-1, 1] with the sign and the zeros kept "
                 "(sklearn MaxAbsScaler); one object over every column that names it")
lego("/pre/sklearn/robust_scaler", "scalers:RobustScaler", state=True, alias="robust_scaler", grouped=True,
     description="Center a column on its median and scale it by the distance between the low and high percentiles "
                 "(sklearn RobustScaler); outliers do not move the statistics")

lego("/pre/sklearn/quantile_transformer", "transformers:QuantileTransformer", state=True, alias="quantile_transformer",
     description="Map a column onto its own quantiles, uniform or normal (sklearn QuantileTransformer); flattens any "
                 "shape, the inverse interpolates between the stored quantiles")
lego("/pre/sklearn/power_transformer", "transformers:PowerTransformer", state=True, alias="power_transformer",
     description="Yeo-Johnson (or Box-Cox for positive columns) with the exponent fitted per column, then "
                 "standardized (sklearn PowerTransformer); the invertible way to a near normal column")
lego("/pre/sklearn/kbins_discretizer", "transformers:KBins", state=True, alias="kbins_discretizer",
     description="Cut a column into bins and write them as one hot columns <field>_bin<n> (encode: ordinal for one "
                 "integer column); strategy quantile, uniform or kmeans (sklearn KBinsDiscretizer)")
lego("/pre/sklearn/spline_transformer", "transformers:Spline", state=True, alias="spline_transformer",
     description="A B-spline basis of a column, <field>_spline<n>: a smooth non linear expansion of one feature that "
                 "a linear head can use (sklearn SplineTransformer)")
