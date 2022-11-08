# -*- coding: utf-8 -*-
"""
The methods and convenience methods to implement hierarchical modeling using pymc
The Poisson/Gamma model is used. The priors are placed on the hyperparameters of the Poisson lambda
parameter:

- y ~ Poisson(lambda)
- lambda ~ Gamma(alpha, beta)
- alpha ~ Gamma(prior_alpha, prior_beta)
- beta ~ Exponential(2)

The results of the inference are divided into three separate forms:

- Posterior predictive (dict): keys=Codes of interest, values=the posterior predictive samples
  for all the sensors.
- Inference Data and Models (dict): keys=Sensors, values=the inference data for the codes from the sensor
- Summary (pd.DataFrame) : The 97% HDI of the parameters for all sensors and codes

questions roger@hammerdirt.ch
"""

import numpy as np
import collections
import pandas as pd
import arviz as az
import dcs
import pymc as pm
from pymc import Gamma, Poisson, Exponential, Model

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# not all locations have prior data. This prior is the same as
# saying the average expected value for an object is 1 for every
# 100 meters with a alot of variance
ref_prior = (1, 100)


def get_summary(data: az.data = None) -> pd.DataFrame:
    """Returns the trace summary from az.summary

    Args:
        data (az.data): The inference data from a model

    Returns:
        The results of az.summary
    """
    return az.summary(data)


def prior_has_feature_data(data: pd.DataFrame = None, feature: str = None, feature_name: str = None) -> bool:
    """Determines whether prior data exists for the feature. Utility method for other functions.

    Args:
        data (pd.DataFrame): The dataframe of prior data
        feature (str): The type of location (river, lake, park, survey area)
        feature_name: The name of the feature on interest
    Returns:
        Boolean, True if there is data, False otherwise
    """

    return feature_name in data[feature].unique()


def make_regional_priors(
        prior_data: pd.DataFrame = None, labels_of_interest: [] = None, value_of_interest: str = "pcs_m",
        column: str = "code", ref_prior: () = ref_prior) -> {}:
    """Returns the mean and standard deviation of each label in column

    For sensors that do not have prior data. The regional results for that
    item are used instead. Values that have a prior mean of 0 receive the
    reference prior. FOR CODES NOT USED PRIOR TO 2020 the reference prior is used.

    The reference prior accounts for the situation where even though the object was
    not identified at a specific sensor it was identified in the same body of water.
    There is some chance that the object may be found.

    Args:
        prior_data (pd.DataFrame, np.ndarray): The data used to define the prior
        column (str): The column name of the categorical variable
        labels_of_interest: The categorical variables of interest
        value_of_interest: The name of the column that holds the numerical value of interest
        ref_prior: A tuple that is used as the shape, scale parameters of the Gamma

    Returns:
        A dictionary of the prior mean and std of the objects of interest for a feature or sensor or both
    """

    if labels_of_interest is None:
        labels_of_interest = prior_data[column].unique()

    regional_priors = {code: [] for code in labels_of_interest}

    for code in labels_of_interest:
        if code in prior_data.code.unique():
            pmask = (prior_data.code == code)
            a_p = prior_data[pmask].copy()
            pmean, pstd = np.mean(a_p[value_of_interest].values), np.std(a_p[value_of_interest].values)

            if pmean == 0:
                pmean, pstd = ref_prior
            if pstd == 0:
                pstd = pmean * 2
        else:
            pmean, pstd = ref_prior

        regional_priors[code] = pmean, pstd

    return regional_priors


def samples_and_priors_for_one_feature(
        samples: pd.DataFrame = None, priors: pd.DataFrame = None, feature: str = None, feature_name: str = None,
        codes: [] = None, default_column: str = "code") -> ():

    """Assembles the current data, prior data and identifies the sensors that correspond to the requested inference

    This method can be run independently or as a component of a loop. If not using the default column, makes sure
    the values in codes are present in the new column.

    Args:
        samples (pd.DataFrame): The samples from the most recent period
        priors (pd.DataFrame): Data from previous surveys
        feature (str): A column name that holds categorical variables
        feature_name (str): A value that can be found in the feature column
        codes ([]): The objects of interest
        default_column (str): The column that holds values for codes

    Returns:
         A data frame of the observed values, the prior values and the sensors of
         interest.
    """

    prior_data = priors[(priors[feature] == feature_name)].copy()
    prior_data = prior_data[prior_data[default_column].isin(codes)]
    observed = samples[samples[feature] == feature_name].copy()
    sensors = observed.sensor.unique()

    return observed, prior_data, sensors


def get_characteristics_of_sensors(data: pd.DataFrame = None, sensors: [] = None) -> pd.DataFrame:
    """Returns the geodata that describes the sensor location

    Args:
        data (pd.DataFrame): The data frame that holds the sensor descriptions
        sensors ([]): The names of the sensors of interest
    Returns:
        A data frame with the sensors of interest
    """

    return data[data.sensor.isin(sensors)]


def chance_of(x, data) -> float:
    """Returns the ratio of the number of values less than x in an array.

    Use this on the posterior samples to return the mass of predicted results
    below a certain threshold.

    Args:
        x (int): A threshold value
        data ([]): A list of values where the threshold is going to be applied
    Returns:
        The ratio  of the number of values less than x to the number of values.
    """
    y = sorted(data)
    achance = len([j for j in y if j < x])
    return achance / len(y)


def get_most_common_and_aggregated_feature_report(
        report_name: str = "Brienzersee", pg: str = "feature",  pgn: str = "brienzersee") -> object:
    """Uses the dcs module to collect the sensors and data for the feature of interest.

    The most common codes for a feature from the most recent sampling period are the default objects of inference.

    Args:
        report_name (str): The name to appear at the top of the report and in Meta data
        pg (str): Parent group, the type of feature that the sensors belong to (river, lake, park)
        pgn: Parent group name, the name of the parent group
    Returns:
        A report class object from the dcs module and the most common codes from the parent group.
    """
    report_args = dict(report_name=report_name, pg=pg, pgn=pgn)
    a_report = dcs.make_this_report(**report_args)
    a_report.set_report_most_common_codes()
    codes = list(a_report.the_most_common_codes)

    return a_report, codes


def separate_local_and_regional_posteriors(inferences: az.data.InferenceData = None, codes: [] = None) -> ():
    """Utility method that creates two data packages and a summary of the posterior for the parameters

    The first data package is a dictionary of the posterior predictive samples from all sensors
    combined by code. The second dictionary is the individual models and inference data for each
    object at each location. The third is a dataframe that contains the posterior results for all
    parameters from all sensors and codes.

    Args:
        inferences (arviz.data.inference_data): An array of inference data
        codes ([]): The objects of interest
    Returns:
        Two dictionaries and a dataframe. One dictionary has all the posterior predictive samples
        from all sensors grouped by code. The second dictionary has the results per sensor, including
        the posterior predictive. The dataframe contains the 97% hdi for each parameter and code at each
        location.
    """
    bsee = {}
    # unpacking inference data results to dict
    for x in inferences:
        for k, v in x.items():
            bsee.update({k: v})

    bsee_keys = bsee.keys()
    # The parameter posteriors for each sensor
    # as well as the posterior predictions for the
    # sensor.
    posteriors = {k: {} for k in bsee_keys}
    # The samples will be used at the regional
    # The posteriors predictions from all sensors
    # for all codes are combined
    inf_d = {x: [] for x in codes}
    # collect all the summaries from all
    # the locations after they have been
    # labeled.
    summaries = []

    for akey in bsee_keys:
        data = bsee[akey]
        for k, v in data.items():
            alpha = v[1].posterior["alpha"]
            beta = v[1].posterior["beta"]
            lmbda = v[1].posterior["lambda"]
            y = np.ravel(v[-1].posterior_predictive["found"])
            amean = np.mean(y)
            v[-2]["sensor"] = akey
            v[-2]["code"] = k
            v[-2]["predicted"] = amean
            inf_d[k].append(y)
            posteriors[akey].update(
                {k: {
                    "alpha": alpha,
                    "beta": beta,
                    "lambda": lmbda,
                    "y": v[-1],
                    "summary": v[-2]}})
            summaries.append(v[-2])

    sx = pd.concat(summaries)
    sx.reset_index(inplace=True)
    sx.rename(columns={"index": "param"}, inplace=True)

    return posteriors, inf_d, sx





def make_data_for_inference(
        feature: str = None, feature_name: str = None, data: pd.DataFrame = None, prior_data: pd.DataFrame = None,
        regional_priors: {} = None, code: str = None, column: str = "pcs_m") -> ():
    """Collects prior data and current observations for feature_name.

    If prior data exists for a code at specific sensor it is used. If no prior data exists for the sensor
    but there is prior data for the region then the regional prior is applied. In the case where the code
    was not used in prior projects a reference prior is assigned. The reference prior of  is the
    same as saying that the average was one piece every hundred meters meters but highly variable.

    Args:
        feature (str): A column that holds different feature names
        feature_name (str): The feature of interest
        data (pd.DataFrame): The most recent project
        prior_data (pd.DataFrame): The results from previous projects
        regional_priors ({}): The priors for all codes at the parent level
        code (str): The object of interest
        column (str): The numeric value "y" or the sample results
    Returns:
        The observed data "y" and the prior parameters for the object
    """

    if prior_data is None or len(prior_data) == 0:
        # if there is no prior data or the length of
        # the prior data for this code is 0 the regional
        # priors are assigned.
        print("making data no prior -> assigning regional prior")
        print(code)
        prior_on_alpha = regional_priors[code]
    else:
        # otherwise collect the mean and standard deviation
        # for the code at this feature/sensor
        pmask = (prior_data[feature] == feature_name)
        prior = prior_data[pmask].copy()
        pmean, pstd = np.mean(prior[column].values), np.std(prior[column].values)
        print("making data prior data present")
        # if the mean is zero of the prior data use
        # the reference prior. If the code is on the list
        # that means it was found in the region, on the same
        # body of water.
        if pmean == 0:
            print("assigning the regional prior")
            pmean, pstd = regional_priors[code]
            # This can happen when there is only one sample and
        # fillna is used. For locations with one sample 2*mean is
        # used as the std.
        if pstd == 0:
            pstd = pmean * 2

        prior_on_alpha = (pmean, pstd)

    # collect the observations from the most recent project
    new_mask = (data[feature] == feature_name)
    obs = data[new_mask].copy()
    print(f"Inference data from : {feature} - {feature_name}")
    print(f"number of observations: {len(obs)}")

    return obs[column].values, prior_on_alpha


def make_model(observed: [] = None, prior_on_alpha: () = None, prior_on_beta: int = 2):
    with Model() as pgamma:
        # the mean and standard deviation of other projects in the region
        mu_prior, sigma_prior = prior_on_alpha
        # are used as the shape and scale parameters
        # of the alpha parameter of lambda
        # link_alpha = (mu_prior**2/sigma_prior**2)
        # link_beta = (mu_prior/sigma_prior**2)
        alpha = Gamma("alpha", alpha=sigma_prior, beta=mu_prior)
        beta = Exponential("beta", lam=prior_on_beta)
        # the proposed posisson rate
        lmbda = Gamma("lambda", alpha=alpha, beta=beta)
        # the posterior
        Poisson("found", mu=lmbda, observed=observed)
        sampled = pm.sample(1000, tune=4000)
        pst = pm.sample_posterior_predictive(sampled)

    return pgamma, sampled, pst


def model_one_code_one_feature(
        feature: str = None, feature_name: str = None, code: str = None, data: pd.DataFrame = None,
        prior_data: pd.DataFrame = None, regional_priors: {} = None, column: str = "pcs_m"):
    print(f"Configuring the prior for {code} at {feature_name}")

    if prior_has_feature_data(data=prior_data, feature=feature, feature_name=feature_name):
        print("has prior data")
        o, p = data[data.code == code].copy(), prior_data[prior_data.code == code].copy()
        args = dict(
            feature=feature,
            feature_name=feature_name,
            data=o,
            prior_data=p,
            code=code,
            regional_priors=regional_priors,
            column=column
        )
        obs, prior_on_alpha = make_data_for_inference(**args)
    else:
        print("has no prior data")
        o = data[data.code == code].copy()
        args = dict(
            feature=feature,
            feature_name=feature_name,
            data=o,
            regional_priors=regional_priors,
            code=code
        )
        obs, prior_on_alpha = make_data_for_inference(**args)

    print(f"This is the prior on Alpha {prior_on_alpha}")

    the_model, sampled, pst = make_model(obs, prior_on_alpha=prior_on_alpha)
    summary = get_summary(sampled)

    return the_model, sampled, summary, pst


def make_inferences(observed: pd.DataFrame = None, prior_data: pd.DataFrame = None, regional_priors: {} = None,
                    sensors: [] = None, feature: str = None, codes: [] = None) -> ():
    """Makes a Pymc model, arviz data object and summary of the results of the variable of interest

    Neither the posterior-predictive nor the pior-predictive are drawn. The summary is intended to be combined with the
    summaries of other models in the generator

    Args:
        observed (pd.DataFrame): The sample data
        prior_data (pd.DtaFrame): Samples of the same region using the same protocol
        regional_priors (dict) : The priors to use is there are none for an object
        sensors: The name of the sensors that observed the data
        codes: The labels for the values of interest
        feature: Usually this is sensor (the originator of the observations)
    Returns:
        generator: The pymc model, the arviz inference data and a summary
    """

    for sensor in sensors:
        res = {sensor: {code: [] for code in codes}}
        for code in codes:
            args = dict(
                feature=feature,
                feature_name=sensor,
                code=code,
                regional_priors=regional_priors,
                data=observed,
                prior_data=prior_data
            )
            res[sensor][code] = model_one_code_one_feature(**args)
        yield res


def get_the_mean_posterior_prediction_for_a_sensor(posteriors, sensor, codes):
    """Gets the average prediction for each code from the posterior predictions for a sensor

    Args:
        posteriors ({}): The dictionary of posteriors after running the model
        sensor (str): The name of the sensor
        codes ([]): The codes of interest

    Returns:
        A dictionary key=Sensor, values = a dictionary keys=code, values=mean
    """
    predictions = {code: 0 for code in codes}
    data = posteriors[sensor]
    for code in codes:
        predictions[code] += np.mean(data[code]["y"].posterior_predictive["found"].values)
    return {sensor: predictions}


def the_mean_posterior_prediction_for_all_sensors_and_codes(posteriors, sensors, codes):
    """Aggregates the posterior predictive of each code for all sensors"""
    predictions = {}
    for sensor in sensors:
        d = get_the_mean_posterior_prediction_for_a_sensor(posteriors, sensor, codes)
        predictions.update(d)
    return predictions


def the_predictions_and_97p_hdi_for_a_location(predictions, summaries, asens, cols, param):
    """A convenience method to display the 94% hdi for a sensor"""
    f = pd.DataFrame(predictions.loc[asens]).reset_index()
    f.rename(columns={"index": "code"}, inplace=True)
    f["hdi_3%"] = 0
    f["hdi_97%"] = 0
    ds = summaries[(summaries.sensor == asens) & (summaries.param == param)][cols]

    f["hdi_3%"] = f.code.map(lambda x: ds.loc[ds.code == x][cols[1]].values[0])
    f["hdi_97%"] = f.code.map(lambda x: ds.loc[ds.code == x][cols[2]].values[0])

    return f


def the_predictions_for_all_codes_and_locations(predictions, summaries, sensors, cols, param):
    """A convenience method to collect all the mean predictions and HDI for all sensors"""
    for asens in sensors:
        pred = the_predictions_and_97p_hdi_for_a_location(predictions, summaries, asens, cols, param)
        yield pred

