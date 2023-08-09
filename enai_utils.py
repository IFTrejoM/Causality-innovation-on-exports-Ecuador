import pandas as pd
from dowhy.causal_model import CausalModel
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Probit

# Function to flatten dictionaries:
def flatten_dict(data_dict):
    """
    The function takes a dictionary as input, it iterates over the key-value pairs of the dictionary, and for each pair, it takes       the key and an item of the list value and adds them to a tuple. This tuple is then added to a list of tuples, which is returned     as the output of the function.
    
    Parameters:
    data_dict (dict): The input dictionary with key-value pairs where the value is a list of items.
    
    Returns:
    list: A list of tuples where each tuple contains a key from the input dictionary and one of its corresponding items from the         list value.
    
    Example:
    >>> data_dict = {'A':[1, 2], 'B':[3, 4], 'C':[5]}
    >>> flatten_dict(data_dict)
    [('A', 1), ('A', 2), ('B', 3), ('B', 4), ('C', 5)]
    """
    items = []
    for key, value_list in data_dict.items():
        for value in value_list:
            items.append((key, value))
    return items

# Function to replace values in pandas.Series
def replace_values_in_series(tuples_list, series):
    """
    The function takes a list of tuples and a pandas series as inputs. It iterates over the list of tuples comparing the second         value of each tuple with the values of the pandas series. When a match is found, the function replaces the value in the series       with the first value of the tuple. The modified series is then returned as the output of the function. This function can be         useful in cases where you have a set of values you want to replace in a pandas series and you have the replacement values in a       list of tuples.
    
    Parameters:
    tuples_list (list): A list of tuples, where each tuple contains a value to be replaced and a replacement value.
    series (pandas.core.series.Series): The input pandas series where the values will be replaced.
    
    Returns:
    pandas.core.series.Series: A pandas series where the values that match the first element of the tuples have been replaced by the     second element of the tuples.
    
    Example:
    >>> tuples_list = [('A', 1), ('B', 2), ('C', 3)]
    >>> series = pd.Series([1, 2, 3, 4, 5])
    >>> replace_values_in_series(tuples_list, series)
    0    A
    1    B
    2    C
    3    4
    4    5
    dtype: object
    """ 
    series = series.replace({x[1]: x[0] for x in tuples_list})
    return series

# Sums values through rows, ignoring NaN-like values
def sum_row_values(df: pd.DataFrame) -> pd.Series:
    """
    This function takes a pandas DataFrame as input and applies a lambda function to each row.
    The lambda function checks if all the values in the row are None, if they are it returns None.
    If there is at least one non-null value but not all, it returns the sum of non-null values.
    If there are no null values in the row, then the function returns the sum of all the values in that row.
    The function returns a pandas Series containing the sum of non-null values of all rows for which at least one non-null value is     present, None if all values are None, and the sum of all values of all rows for which there are no null values.

    Parameters:
    df (pd.DataFrame): DataFrame to be processed

    Returns:
    pd.Series: Series containing the sum of non-null values of rows for which at least one non-null value is present, None if all       values are None, and the sum of all values of all rows for which there are no null values.
    """
    result = df.apply(lambda x: None if x.isnull().all() else (x.sum(skipna=True) if x.isnull().any() else x.sum()), axis=1)
    return result

# Function to compute the Inverse Propensity Weighted (IPW) Causal Estimate for input data:
def ipw_causal_estimate(data, treatment, outcome, common_causes):
    """
    Compute the Inverse Propensity Weighted (IPW) Causal Estimate for input data.

    Parameters:
    data (pandas DataFrame): Data containing the treatment, outcome and common causes.
    treatment (string): Name of the column in data representing the binary treatment assignment.
    outcome (string): Name of the column in data representing the outcome variable.
    common_causes (list of strings): List of names of columns in data representing the common causes.

    Returns:
    float: IPW Causal Estimate.

    Raises:
    ValueError: If the input data does not meet the required specifications.

    Example:
    data = pd.read_csv("data.csv")
    treatment = "Treatment"
    outcome = "Outcome"
    common_causes = ["Cause1", "Cause2"]
    ipw_causal_estimate(data, treatment, outcome, common_causes)
    """
    # 1) Instantiate a CausalModel object:
    model = CausalModel(data=data, treatment=treatment, outcome=outcome, common_causes=common_causes)

    # 2) Identify causal effect and return target estimands:
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # 3) Estimate the target estimand using a IPW:
    causal_estimate_ipw = model.estimate_effect(identified_estimand,
                                                    method_name="backdoor.propensity_score_weighting",
                                                    target_units = "att",
                                                    method_params={"weighting_scheme":"ips_stabilized_weight"})
    # 4) Return IPW causal estimate:
    return causal_estimate_ipw.value

# Function to fit a Probit regression model to the given data:
def get_probit_model(X, y, var_weights=None, method=None, cov_type=None):
    """
    Fits a Probit regression model to the given data.

    Parameters:
    -----------
    X : array-like
        The independent variables.
    y : array-like
        The dependent variable.
    var_weights : array-like or None, optional
        The weights for each observation. If None, equal weights are assumed.
    method : str or None, optional
        The optimization method to use. If None, the default optimization method 
        for the chosen family is used.
    cov_type : str or None, optional
        The type of covariance matrix estimator to use. If None, the default 
        estimator for the chosen family and optimization method is used.

    Returns:
    --------
    summary : str
        A summary of the fitted Probit regression model.
    """
    # Create an instance of the Probit link function
    probit_instance = Probit()
    
    # Fit a generalized linear model with a binomial family and the Probit link function
    probit_model = GLM(y, sm.add_constant(X),
                       var_weights=var_weights,
                       family=Binomial(link=probit_instance)
                      ).fit(method=method, cov_type=cov_type)
    
    # Return the summary of the fitted model
    return probit_model