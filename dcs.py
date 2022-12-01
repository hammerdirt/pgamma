# -*- coding: utf-8 -*-
import collections
import csv
from dataclasses import dataclass, field
import datetime as dt
import time
import numpy as np
import scipy.stats
from scipy.stats import beta
from scipy.stats import bernoulli
from collections import Counter
import inspect
import functools

import pandas as pd
from slugify import slugify
import random

import priors
from fractions import Fraction


import config

# config.start_log()


def inspector() -> (str, int):

    lineno = inspect.stack()[1].lineno
    who = inspect.stack()[1].function
    context = inspect.stack()[1].code_context[0].strip(" ")

    return lineno, who, context


class CodeDataClass:
    """
    Handles the definition of the <code> values. Maps the code to attributes of the object:

    - description
    - material
    - groupname

    The CodeDataClass is used to make the column and row labels for charts, calculate the
    quantity as a function of material and aggregate codes according to use.
    """
    def __init__(self):
        self.label = "Code attributes"
        self.data = config.retrieve_data(source="new_codes.csv", dtype=config.code_data(), ext="csv")
        self.description = {x["code"]: x["description"] for x in self.data}
        self.material = {x["code"]: x["material"] for x in self.data}
        self.groupname = {x["code"]: x["groupname"] for x in self.data}


class SensorDetails:
    """Handles the access to the geographic and demographic variables for a Sensor
    """

    def __init__(self):
        self.label = "Code attributes"
        self.data = config.retrieve_data(source="dc_locations.csv", dtype=config.sensor_columns(), ext="csv")
        self.place_names_map = None

    def set_place_names_map(self):
        unique_location_names = np.unique(self.data["location"])
        location_names = dict(zip([slugify(x) for x in unique_location_names], unique_location_names))
        unique_city_names = np.unique(self.data["city"])
        city_names = dict(zip([slugify(x) for x in unique_city_names], unique_city_names))
        unique_feature_names = np.unique(self.data["feature"])
        water_names = dict(zip([slugify(x) for x in unique_feature_names], unique_feature_names))
        unique_survey_areas = np.unique(self.data["survey_area"])
        survey_area = dict(zip([slugify(x) for x in unique_survey_areas], unique_survey_areas))

        place_names_map = {
            "sensor": location_names,
            "city": city_names,
            "feature": water_names,
            "survey_area": survey_area
        }

        self.place_names_map = place_names_map


def code_description_map(codes: dict = None, needs_description: list = None, value: str = None, lang: str = None) -> []:
    """Given a list of codes this returns the description of the object"""

    if value in ["% of total group", "group_pcs_m"]:
        descriptions = [f"{x[0].upper()}{x[1:]}" for x in needs_description]
    else:
        descriptions = [codes[x] for x in needs_description]

    return descriptions


available_reports = [
    "make_table_survey_total_summary",
    "make_summary_table_of_child_activities",
    "make_report_code_summary",
    "make_report_scatter_chart-readme",
    "make_cumulative_distribution_chart",
    "make_fragmented_plastics_table",

]


def text_translate_map(lang: str = None, figure_id: str = None) -> list[str]:

    a, b, c = inspect.stack()[1].lineno, inspect.stack()[1].function, inspect.stack()[1].code_context[0].strip(" ")
    line = f"TEXT FORMATTER {a}, {b}, context: {c}"

    if lang is None:
        lang = config.language
    else:
        pass

    if figure_id in available_reports:
        return config.text_maps_for_standard_figures[figure_id][lang]
    else:
        some_text = "The figure id did not match"
        config.append_to_log(f"{line}: result {some_text}\n")
        raise ValueError("There was no match")


def local_date_formatter(lang: str = None):
    a, b, c = inspector()
    line = f"DATE FORMATTER {a}, {b}, context: {c}"

    with open(config.current_log_file, "a") as a_file:
        a_file.write(f"{line} result {config.date_format(lang)}\n")

    pass


def local_numeric_formatter(lang: str = None, needs_formatting: list = None):
    a, b, c = inspector()
    line = f"NUMERIC FORMATTER {a}, {b}, context: {c}"

    if lang == "en":
        formatted = ['{:,}'.format(x) for x in needs_formatting]
    else:
        formatted = ['{:,}'.format(x).replace(',', ' ') for x in needs_formatting]

    with open(config.current_log_file, "a") as a_file:
        a_file.write(f"{line} result: !  !\n")

    return formatted


def row_and_column_label_formatter(lang: str = None, rows_columns: dict = None, place_names_map: dict = None, codes: np.ndarray = None) -> ():
    a, b, c = inspector()
    inspecting = f"ROW and LABEL FORMATTER {a}, {b}, context: {c}"

    if place_names_map is None:
        place_names_map = SensorDetails()
        place_names_map.set_place_names_map()

    assert lang is not None, "No language provided"

    # maps and keys
    place_names_map = place_names_map.place_names_map
    these_keys = list(rows_columns.keys())

    # the resulting package
    row_column_labels = {}

    for key in these_keys:
        if key == "code":
            if codes is None:
                c = CodeDataClass()
                codes = c.description
            use = rows_columns[key]["use"]
            formatted = code_description_map(codes=codes, needs_description=rows_columns[key]["values"])
            row_column_labels.update({use: formatted})

        elif key == "place":
            use = rows_columns[key]["use"]
            group = rows_columns[key]["level"]
            assert isinstance(rows_columns[key]["values"], list), "this needs to be a list"
            slugged = [slugify(x) for x in rows_columns[key]["values"]]
            formatted = [place_names_map[group][x] for x in slugged]
            row_column_labels.update({use: formatted})

        elif key == "number":
            use = rows_columns[key]["use"]
            vals = rows_columns[key]['values']
            # if the values are an nd array d >= 2
            if use == "field":
                assert isinstance(rows_columns[key]["values"][0], list), "this needs to be a list"
                formatted = [local_numeric_formatter(lang=lang, needs_formatting=x) for x in vals]
                row_column_labels.update({use: formatted})
            else:
                assert isinstance(rows_columns[key]["values"], list), "this needs to be a list"
                formatted = local_numeric_formatter(lang=lang, needs_formatting=vals)
                row_column_labels.update({use: formatted})
            pass

        elif key == "date":
            use = rows_columns[key]["use"]
            vals = rows_columns[key]['values']
            assert isinstance(rows_columns[key]["values"], list), "this needs to be a list"
            formatted = [local_numeric_formatter(lang=lang, needs_formatting=x) for x in vals]
            row_column_labels.update({use: formatted})

        elif key == "text":
            use = rows_columns[key]["use"]
            figure = rows_columns[key]["figure_id"]
            formatted = text_translate_map(figure_id=figure)
            row_column_labels.update({use: formatted})

        elif key in ["group_pcs_m", "% of total group", "% of total", "pcs_m"]:
            use = rows_columns[key]["use"]
            value = rows_columns[key]["value"]
            codes = CodeDataClass().description
            if use == "row_labels":
                formatted = code_description_map(lang=lang, codes=codes, value=value, needs_description=rows_columns[key]["values"])
                row_column_labels.update({use: formatted})

        else:
            result = "caused value error"
            config.append_to_log(line=f"{inspecting} result: {result}")

            raise ValueError("The provided dictionary had no matching keys")

    result = "objects formatted"
    config.append_to_log(line=f"{inspecting} result: {result}")

    return row_column_labels


@dataclass
class Thing:
    """A Thing is the quantity and rate at which an object was found

    Objects are defined by codes. The codes are listed in the Guide
    to monitoring litter on European seas. The CodeData class contains
    the complete description and material composition for each code.
    """
    code: str = None
    group: str = None
    quantity: int = None
    pcs_m: float = None


@dataclass
class Survey:
    """A Survey is a collection of Things from one place on one day.

    The instance is the unique identifier for the Survey. Sensor is the
    identifier for the sensor. City, feature, survey area are different
    hierarchical groupings of the data. A survey is further defined by this
    combination of groupings.
    """
    instance: str = None
    sensor: str = None
    city: str = None
    feature: str = None
    survey_area: str = None
    date: str = None
    things: list[Thing] = field(default_factory=list)

    def survey_quantity(self) -> int:
        """The sum of quantity or number of objects at the survey"""
        q = [x.quantity for x in self.things]
        assert sum(q) >= 0, "This should be at least zero"
        return sum(q)

    def survey_pcs_m(self) -> float:
        """The sum of pcs/m of all objects at the survey"""
        pcs = [x.pcs_m for x in self.things]
        assert sum(pcs) >= 0, "This should at least 0"
        return np.round(sum(pcs), 3)

    def code_quantity(self) -> dict:
        """The quantity found of each thing"""
        assert isinstance(self.things[0], Thing), "These should be data class instances"
        data = {x.code: x.quantity for x in self.things}
        return data

    def code_pcs_m(self) -> dict:
        """The pcs/m rate for each thing"""
        data = {x.code: x.pcs_m for x in self.things}
        return data

    def code_group_quantity(self) -> dict:
        """The sum of things in the same <group>"""

        a_key = "group"
        things = self.things
        a_func = sum
        kwargs = dict(a_method_name="quantity", is_callable=False)

        result = config.collect_methods(things=things, a_func=a_func, a_key=a_key, **kwargs)

        return result

    def code_group_pcs_m(self):
        """The sum of things in the same <group>"""

        a_key = "group"
        things = self.things
        a_func = sum
        kwargs = dict(a_method_name="pcs_m", is_callable=False)

        result = config.collect_methods(things=things, a_func=a_func, a_key=a_key, **kwargs)

        return result


class Sensor:
    """A Sensor is an aggregator of Surveys. It represents the observations
    within a defined geographic range.

    In addition to the labels from Surveys the Sensor adds the member attribute. The
    member attribute designates which child group the sensor belongs to it is assigned
    according to the user selection and the existing relationship.

    The sensor class provides access to the aggregated survey results for all surveys
    executed by the sensor.
    """

    def __init__(self, sensor: str = None, member: str = None, surveys: [] = None):
        self.sensor = sensor
        self.member = member
        self.feature = None
        self.city = None
        self.survey_area = None
        self.surveys = surveys
        self.tries_fails = None
        self.tfdf = None

    def set_membership(self, member_of: str = None):
        """Called by the constructor function once membership is defined"""
        self.member = member_of

    def set_sensor_city_feature_survey_area(self, labels: dict = None) -> None:
        """Assigns the grouping criteria to the sensor, called by the constructor

        The attributes of the class are assigned by the constructor that instantiates it
        """
        attribute_key = {
            "survey_area": self.survey_area,
            "city": self.city,
            "feature": self.feature
        }
        for label in labels:
            attribute_key[label] = labels[label]

    def sensor_survey_results(self) -> []:
        """The total in pcs_m for each survey that the sensor is responsible for

        The totals are collected from each survey class in the sensor

        :return: A time series of survey totals in pcs/m
        """
        return [(x.date, x.survey_pcs_m()) for x in self.surveys]

    def sensor_median(self) -> np.ndarray:
        """The median survey result in things/m"""
        return np.median([x.survey_pcs_m() for x in self.surveys])

    def sensor_average(self) -> np.ndarray:
        """The average things/m"""
        return np.mean([x.survey_pcs_m() for x in self.surveys])

    def sensor_quantity(self) -> np.ndarray:
        """The total number of things for this sensor"""
        return np.sum([x.survey_quantity() for x in self.surveys])

    def sensor_nsamps(self) -> int:
        """The number of surveys from this sensor"""
        return len(self.surveys)

    def sensor_date_range(self) -> ():
        """The first and last recorded survey"""
        string_dates = np.array([x.date for x in self.surveys]).astype(dt.date)
        start = string_dates.min()
        end = string_dates.max()
        return start, end

    def sensor_code_totals(self) -> dict:
        """The total for each Thing in inventory"""
        s = [Counter(x.code_quantity()) for x in self.surveys]
        totals = sum(s, Counter())
        return totals

    def sensor_code_results_qty(self) -> dict:
        """Collect the individual results for each code into an array
        The results are stored in a dictionary where the key = code.
        """
        assert isinstance(self.surveys[0].code_quantity(), dict)
        results = {k: [v] for k, v in self.surveys[0].code_quantity().items()}

        for a_survey in self.surveys[1:]:
            for k, v in a_survey.code_quantity().items():
                results[k].append(v)

        return results

    def sensor_code_results_pcs_m(self) -> dict:
        """Collect the individual results for each code into an array
        The results are stored in a dictionary where the key = code.
        """
        assert isinstance(self.surveys[0].code_pcs_m(), dict)
        results = {k: [v] for k, v in self.surveys[0].code_pcs_m().items()}

        for a_survey in self.surveys[1:]:
            for k, v in a_survey.code_pcs_m().items():
                results[k].append(v)

        return results

    def sensor_code_group_totals(self) -> dict:
        """Collect the individual results for each code-group into an array
        The results are stored in a dictionary where the key = code.
        """
        results = collections.Counter({})
        for survey in self.surveys:
            results += collections.Counter(survey.code_group_quantity())
        return results

    def sensor_code_group_pcs_m(self) -> dict:
        """Collect the individual results for each code-group into an array
        The results are stored in a dictionary where the key = code.
        """
        results = {}

        for code in list(self.surveys[0].code_group_quantity().keys()):
            results.update({code: []})
            for survey in self.surveys:
                results[code].append(survey.code_group_pcs_m()[code])

        return results

    def sensor_tries_fails(self) -> None:
        """Tests the condition X > i where x = the survey result for a code and i is an integer value > 0.
        The limit of i is defined by the <fails> variable in config

        Updates the attribute of the class that calls it.
        """
        keys = config.dictionary_keys_for_fails()
        tries_fails = {code: {0: [0, 0]} for code in self.sensor_code_results_qty()}
        this_code_data = self.sensor_code_results_qty()

        for code in this_code_data:
            code_data = this_code_data[code]
            this_limit = np.max(code_data)
            for i in np.arange(start=0, stop=this_limit+1, step=1):
                vals = [x for x in code_data if x > i]
                this_key = i
                if this_key in tries_fails[code].keys():
                    tries_fails[code][this_key] = np.array([len(vals), len(code_data)])
                else:
                    tries_fails[code].update({this_key: np.array([len(vals)+1, len(code_data)+1])})

        self.tries_fails = tries_fails

    def __str__(self):
        a_string = f"\nSensor={self.sensor}, member={self.member}, dates={self.sensor_date_range()},\
n samples={self.sensor_nsamps()}, average pcs={self.sensor_average()}, total pcs={self.sensor_quantity()}."

        return a_string

    def __repr__(self):
        a_string = f"\nSensor={self.sensor}, member={self.member}, dates={self.sensor_date_range()},\
n samples={self.sensor_nsamps()}, average pcs={self.sensor_average()}, total pcs={self.sensor_quantity()}."

        return a_string


class ReportBase(object):
    """The ReportBase defines the attributes needed to summarize the survey results. This includes:

    - Time series of survey totals
    - Descriptive statistics at each level of aggregation
    - quantiles
    - mean, median, standard deviation, fail, rate
    - automatic drill down for all child elements

    The base contains no methods for instantiating its own attributes. The constructor function packages
    the surveys and handles the labeling of all the elements.
    """

    def __init__(
            self, report_name: str = None, parent_group_name: str = None, parent_group: str = None,
            child_group: str = None, sensors: list[Sensor] = None,
            lang: str = None, user: str = None, date: str = None
    ):

        self.report_name = report_name
        self.parent_group_name = parent_group_name
        self.parent_group = parent_group  # one of survey area, city, feature or sensor
        self.child_group = child_group  # one of survey area, city, feature or sensor
        self.children = None  # the names of child group elements in the parent group
        self.sensors = sensors
        self.sensor_names = None
        self.lang = lang
        self.user = user
        self.date = date
        self.date_range = None
        self.n_sensors = None
        self.n_samples = 0
        self.quantity = None
        self.survey_totals = None
        self.q_tiles = None
        self.report_code_totals = None
        self.min_max_survey = None
        self.tries_fails = None
        self.code_stats = None
        self.the_most_common_codes = None
        self.code_group_totals = None
        self.code_group_results = None
        self.code_group_p_totals = None


class ReportMethods(ReportBase):
    """Contains the methods needed to instantiate the attributes of the base.

    Report methods provides two services:

    - The graphics and descriptive statistics for each group and subgroup
    - Deliver the survey results to a predictive model

    Methods with the <set> prefix assign values to attributes, those with the <report> prefix
    make available data critical for a calculation but are not intended for display. The <make> prefix
    defines the methods that are creating output for a charting library. Methods destined for display
    are accompanied by a formatting helper for columns, labels and rows. Column and row labels are
    provided for each graphic as well as a readme if needed.
    """

    def set_and_label_children(self, membership_keys: dict = None) -> None:
        """Labels the Sensor objects according to the user requests.

        Called by the constructor function once the labels are defined
        """
        for sensor in self.sensors:
            if membership_keys is None:
                try:
                    assert sensor.member is not None, "This should be set"
                except KeyError:
                    print(f"The Sensor.member attributes is not set and there are no keys")
                    print("Exiting the program")
                    break
            else:
                try:
                    sensor.member = membership_keys[sensor.sensor]
                except KeyError:
                    print(f"The sensor name is not in the keys")
                    print("Exiting the program")
                    break

        self.children = {x.member for x in self.sensors}

    def set_report_date_range(self) -> None:
        """The min and max dates from the Sensor elements"""

        if self.sensors is None:
            self.date_range = None
        else:
            s_dates = np.array([x.sensor_date_range()[0] for x in self.sensors]).astype(dt.datetime)
            assert isinstance(s_dates[0], str), "This should be a str rep of a date"

            e_dates = np.array([x.sensor_date_range()[1] for x in self.sensors]).astype(dt.datetime)
            assert isinstance(e_dates[0], str), "This should not be a str rep of a date"

            self.date_range = min(s_dates), max(e_dates)

    def set_report_number_of_sensors(self) -> None:
        """The number of sensors in the report"""

        assert len(self.sensors) > 0, "There are no sensors loaded"

        self.n_sensors = len(self.sensors)

    def set_report_sensor_names(self) -> None:
        """An array of the sensor names for each child element"""

        assert len(self.sensors) > 0, "There are no sensors loaded"
        assert isinstance(self.sensors[0].sensor, str), "This should be a string"
        self.sensor_names = [x.sensor for x in self.sensors]

    def set_report_number_of_samples(self) -> None:
        """Counts the number of samples from each child element"""

        assert len(self.sensors) > 0, "There are no sensors loaded"
        assert self.sensors[0].sensor_nsamps() > 0, "There cannot be a sensor with no samples"
        assert isinstance(self.sensors[0].sensor_nsamps(), int), "This should be an integer"

        result = sum([x.sensor_nsamps() for x in self.sensors])
        self.n_samples = result

    def set_report_quantity(self) -> None:
        """Calculates the sum of all objects identified"""
        
        assert len(self.sensors) > 0, "There are no sensors loaded"
        assert self.sensors[0].sensor_quantity() > 0, "There cannot be a sensor with zero quantity"
        assert isinstance(self.sensors[0].sensor_quantity(), np.int64), "This should be an integer"

        result = sum([x.sensor_quantity() for x in self.sensors])
        self.quantity = result

    def set_report_survey_totals(self) -> None:
        """Concatenates the survey totals and date from the child elements"""

        assert len(self.sensors) > 0, "There are no sensors loaded"
        assert len(self.sensors[0].sensor_survey_results()) > 0, "There are no survey totals"

        survey_results = [x.sensor_survey_results() for x in self.sensors]
        results = np.concatenate(survey_results)

        assert isinstance(results[0], np.ndarray), "This should be a date, float array"
        assert len(results[0]) == 2, "It should have length two"
        assert isinstance(results[0][0], str), "This should be a string date"
        assert isinstance(results[0][1], str), "This should be a string representation of a float"

        self.survey_totals = results

    def set_report_quantiles(self) -> None:
        """The quantiles, defined in config, of the survey totals"""

        assert len(self.sensors) > 0, "There are no sensors loaded"
        if self.survey_totals is None:
            self.set_report_survey_totals()

        assert isinstance(self.survey_totals[0], np.ndarray), "These should be tuples"
        results = np.quantile([float(x[1]) for x in self.survey_totals], config.quantiles)

        self.q_tiles = results

    def set_report_max_min(self) -> None:
        """The maximum and minimum recorded survey value"""

        assert len(self.sensors) > 0, "There are no sensors loaded"
        if self.survey_totals is None:
            self.set_report_survey_totals()

        assert isinstance(self.survey_totals[0], np.ndarray), "These should be an array"
        
        data = [float(x[1]) for x in self.survey_totals]
        a_min = min(data)
        a_max = max(data)

        self.min_max_survey = a_min, a_max

    def set_report_code_totals(self) -> None:
        """The total number of an object found for all surveys in the report"""

        assert len(self.sensors) > 0, "There are no sensors loaded"
        assert isinstance(self.sensors[0].sensor_code_totals(), dict), "This should be a dict"

        # make each dict a Counter object
        s = [Counter(x.sensor_code_totals()) for x in self.sensors]
        totals = sum(s, Counter())

        assert totals["G27"] >= 0, "This should not be none"

        self.report_code_totals = totals

    def report_code_pcs_m(self) -> dict:
        """The median pcs/m for each code from all surveys in the report"""

        assert len(self.sensors) > 0, "There are no sensors loaded"

        d = [x.sensor_code_results_pcs_m() for x in self.sensors]
        codes = list(d[0].keys())
        results = {}
        for code in codes:
            cd = [x[code] for x in d]
            code_datas = np.concatenate(cd)
            result = {code: np.median(code_datas)}
            results.update(result)

        return results

    def report_code_fail_rate(self) -> dict:
        """Reports the number of times that an object value exceeded a number i for i [0-50]

        Creates a dictionary of tables, one table for each code, that documents the number of times
        that the number found at a survey exceeded a given value. The given value(s) are 0-50 inclusive.
        """

        assert len(self.sensors) > 0, "There are no sensors loaded"
        for x in self.sensors:
            assert x.tries_fails is not None, "These should have been instantiated already"
        assert len(self.sensors[0].tries_fails) > 1, "Whats going on?"

        d = [x.tries_fails for x in self.sensors]

        results = {}
        codes = list(d[0].keys())

        for code in codes:
            success = 0
            fail = 0
            for a_sensor in d:
                fail += a_sensor[code][0][0]
            rate = fail/self.n_samples

            assert success <= self.n_samples, "The number of success cannot be greater than the the number of tries"
            assert fail <= self.n_samples, "The number of failures cannot exceed the the number of tries"
            assert rate <= 1, "This should not exceed 1"
            results.update({code: {"fail": fail, "success": success, "rate": rate}})

        return results

    def set_report_code_group_stats(self) -> None:
        """Set the results by code-group for all the report data in median pcs/m and % of total"""

        t = [x.sensor_code_group_totals() for x in self.sensors]
        totals = sum(t, collections.Counter())
        p_totals = {k: v / self.quantity for k, v in totals.items()}

        pcs_mx = [x.sensor_code_group_pcs_m() for x in self.sensors]
        r = {k: [np.median(v)] for k, v in pcs_mx[0].items()}

        for vals in pcs_mx[1:]:
            for a_group in r.keys():
                r[a_group] += vals[a_group]
        results = {k: np.median(v) for k, v in r.items()}

        self.code_group_p_totals, self.code_group_results = p_totals, results

    def set_report_most_common_codes(self) -> None:
        """identifies the objects most frequently reported and those objects that were the most numerous"""

        # if the report code totals have not been set do it now
        if self.report_code_totals is None:
            self.set_report_code_totals()

        assert self.report_code_totals is not None, "There are no code totals"

        # get the most abundant by quantity
        # the report code totals are stored as a Counter object
        # calling the .most_common will sum all instances of each code
        by_quantity = [x[0] for x in self.report_code_totals.most_common(10)]
        # if there are two or fewer samples return by quantity only
        if self.n_samples <= 2:
            self.the_most_common_codes = by_quantity
        else:
            # the report fail rate gives the results by how many times at least one was found
            by_frequency = self.report_code_fail_rate()
            # check_output(by_frequency)

            # check the <fail> key for each code against the requested fail_rate
            most_frequent = {k: v["fail"] for k, v in by_frequency.items() if v["rate"] >= config.fail_rate}

            # the most common is the set of objects that were either in the 10 most abundant
            # of those objects found in at least one of two surveys.
            the_most_common = set(by_quantity) | set(list(most_frequent.keys()))
            self.the_most_common_codes = the_most_common

    def make_table_survey_total_summary(self) -> (list, dict):
        """Makes the table of descriptive statistics of the report survey totals"""

        assert len(self.sensors) > 0, "There are no sensors loaded"

        labels = [
            "Report name",
            "Date range",
            "N samples",
            "N sensors",
            "N pieces",
            f"Median {config.rate_label[config.alt_rate]}",
            f"Minimum {config.rate_label[config.alt_rate]}",
            f"Maximum {config.rate_label[config.alt_rate]}"
        ]

        values = [
            self.report_name,
            self.date_range,
            self.n_samples,
            self.n_sensors,
            self.quantity,
            self.q_tiles[2],
            self.min_max_survey[0],
            self.min_max_survey[1]
        ]

        if config.language != "en":
            needs_formatting = {"text": {"figure_id": "make_table_survey_total_summary", "values": labels, "use": "column"}}
            labels = row_and_column_label_formatter(lang=self.lang, rows_columns=needs_formatting)

        a_table = [labels, values]

        return a_table, labels

    def make_summary_table_of_child_activities(self) -> (list, dict):
        """Summarizes the number of samples, hours, weights and surface area at each child element"""

        rows = []

        for a_member in self.children:
            data = [x for x in self.sensors if x.member == a_member]
            nsamps = sum([x.sensor_nsamps() for x in data])
            qty = sum([x.sensor_quantity() for x in data])
            s_results = [x.sensor_survey_results() for x in data]
            flattened = [x[1] for j in s_results for x in j]
            median_pcs_m = np.round(np.median(flattened), 3)
            weights, measures, times = config.get_dimensional_data(data)

            a_row = [
                a_member,
                nsamps,
                qty,
                median_pcs_m,
                round(weights[0]/config.weight_denom, 3),
                round(weights[1], 3),
                round(measures[0], 3),
                round(measures[1], 3),
                round(times/60, 2)
            ]
            rows.append(a_row)

        first_row = [
            "Feature",
            "N samples",
            "N pieces",
            "Median pcs/m",
            "Plastic weight",
            "Total weight",
            "N meters",
            "N meters²",
            "N minutes"
        ]

        if config.language != "en":
            needs_formatting = {"text": {"figure_id": "make_summary_table_of_child_activities", "values": first_row, "use": "column"}}
            column = row_and_column_label_formatter(lang=self.lang, rows_columns=needs_formatting)
            first_row = column["column"]

        a_table = [first_row, *rows]

        return a_table

    def make_report_code_summary(self, codes: np.ndarray = None) -> (list, dict):
        """Summarizes the survey results for each code from all the data

        From this table the most abundant and most common objects can be defined. The table defines the following for
        each object:

        - description: The plain english description of the "GCode"
        - qty: The total number of pieces found
        - pcs_m: The median survey result in pcs/m
        - % of total: The percent of the total found
        - fail: The ration of the number of times at least one object was found divided by the number of samples

        :return:
        """

        data_pcs_m = self.report_code_pcs_m()
        data_fails = self.report_code_fail_rate()
        assert isinstance(data_fails, dict), "Not a dictionary"
        assert len(data_pcs_m) > 0, "There is no rate data"
        assert len(data_fails) > 0, "There is no integer data"
        ks = list(data_fails.keys())
        assert data_fails[ks[0]] is not None, "This should trip it up"

        if codes is None:
            c = CodeDataClass()
            codes = c.description

        results = []

        the_codes = list(self.report_code_totals.keys())

        descriptions = code_description_map(codes=codes, needs_description=the_codes)

        assert isinstance(descriptions, list), "This needs to be a list"

        for i, code in enumerate(the_codes):
            # get the code description in the required language
            description = descriptions[i]
            pcs_m = float(round(data_pcs_m[code], 2))
            quantity = int(self.report_code_totals[code])
            percent_total = float(round((quantity/self.quantity), 0))
            f_rate = float(round((data_fails[code]["rate"]), 3))
            a_row = [code, description, quantity, pcs_m, percent_total, f_rate]
            results.append(a_row)

        needs_formatting = {"text": {"figure_id": "make_report_code_summary", "use": "top row", }}

        first_row = row_and_column_label_formatter(lang=self.lang, rows_columns=needs_formatting)

        code_report = [first_row["top row"], *results]

        return code_report

    def make_heat_map_codes(self, value: str = None) -> (list, dict):
        """The most common codes at the parent level are given as the median pcs/m or % of total for each child feature

        The resulting is an nd matrix, the rows are the most common items and the columns are the features. The column
        and row labels are part of the output.

        :return:
        """

        # if the most common codes have not been set
        # do that now
        if self.the_most_common_codes is None:
            self.set_report_most_common_codes()
        if self.code_group_p_totals is None:
            self.set_report_code_group_stats()

        members = list(self.children)
        # the output
        a_table = []
        col = []

        # group the sensors by membership in child group:
        check_total = 0
        for member in members:
            code_labels = list(self.the_most_common_codes)
            member_data = [sensor for sensor in self.sensors if sensor.member == member]
            assert isinstance(member_data[0], Sensor), "This should be a sensor class"
            col.append(member)
            column = []
            if value == "% of total":
                for code in code_labels:
                    cdata = [sensor.sensor_code_totals()[code] for sensor in member_data]
                    assert isinstance(cdata[0], int), "This should be a counter object"
                    new_totals = sum(cdata)
                    check_total += new_totals
                    p_total = round(new_totals/self.quantity, 2)
                    assert p_total < 1, "This is a percent and should be less than one"
                    column.append(p_total)

                assert len(column) == len(code_labels), "this should be a complete column for each location"
                a_table.append(column)
            elif value == "pcs_m":
                these_values = [x.sensor_code_results_pcs_m() for x in member_data]
                assert isinstance(these_values[0], dict), "This should be a sensor class"
                code_labels = list(self.the_most_common_codes)
                for code in code_labels:
                    cd = []
                    for h in these_values:
                        cd.extend(h[code])
                    column.append(np.round(np.median(cd), 2))
                a_table.append(column)
            elif value == "% of total group":
                these_values = [sensor.sensor_code_group_totals() for sensor in member_data]
                new_totals = sum(these_values, collections.Counter())
                check_total += sum(new_totals.values())
                code_labels = list(self.code_group_results.keys())
                for code in code_labels:
                    column.append(round(new_totals[code]/self.quantity, 2))

                a_table.append(column)
            elif value == "group_pcs_m":
                # these_values = [sensor.sensor_code_group_pcs_m() for sensor in member_data]
                these_values = [sensor.sensor_code_group_pcs_m() for sensor in member_data]
                code_labels = list(self.code_group_results.keys())
                for code in code_labels:
                    cd = []
                    for h in these_values:
                        cd.extend(h[code])
                    column.append(np.round(np.median(cd), 2))
                a_table.append(column)

            else:
                raise ValueError(value)

        if value in ["% of total group"]:
            assert check_total == self.quantity, f"""
            
                        !! Stop This is serious. It means there is a problem in the aggregation methods !!
            
            
                       Under these conditions the two values should be the same {check_total} != {self.quantity}
                       
                       
                                                  """

        needs_formatting = {
            value: {
                "values": code_labels,
                "use": "row_labels",
                "value": value
            },
            "place": {
                "use": "column_labels",
                "level": self.child_group,
                "values": members
            }
        }

        formatted = row_and_column_label_formatter(lang=self.lang, rows_columns=needs_formatting)
        first_column = ["Objects", *formatted["row_labels"]]
        # make formatted column and row labels
        finished_table = [first_column]
        if len(a_table) > 1:
            for i, each_column in enumerate(a_table):

                new_column = [formatted["column_labels"][i-1], *each_column]
                finished_table.append(new_column)
        else:
            for d in a_table:
                if isinstance(d, list):
                    new_table = [[first_column[i - 1], x] for i, x in enumerate(d)]
                    finished_table = [*finished_table, *new_table]
                    for k, word in enumerate(d):
                        pass

        return finished_table

    def make_report_scatter_chart(self) -> (list, dict):
        """The time series values of the survey totals"""

        if self.survey_totals is None:
            self.set_report_survey_totals()
        return self.survey_totals

    def make_cumulative_distribution_chart(self) -> (list, dict):
        """The observed distribution of survey results
        """

        if self.survey_totals is None:
            self.set_report_survey_totals()

        vals = np.array(self.survey_totals[:, 1]).astype(float)
        srt = sorted(vals)
        y = [float((i+1)/len(srt)) for i, _ in enumerate(srt)]

        results = np.column_stack([srt, y])

        return results

    def make_fragmented_plastics_table(self) -> (list, dict):
        """
        :return:
        """

        frag_csv = config.retrieve_data(source=config.frag_path, ext="csv")
        c = CodeDataClass()
        codes = c.description

        data_type_frags = [
            ('sensor', 'U100'),
            ('date', 'M8[D]'),
            ('code', 'U7'),
            ('pcs_m', 'f8'),
            ('quantity', 'i4'),
            ('loc_date', 'U100'),
        ]

        fps = np.array([tuple(x) for x in frag_csv], dtype=data_type_frags)

        data = fps[np.isin(fps['sensor'], self.sensor_names)]
        some_codes = np.unique(data["code"])
        samp_s = np.unique(data["loc_date"])

        assert len(samp_s) == self.n_samples, "These should be equal"

        a_table = []
        for code in some_codes:
            d = data[data["code"] == code]
            t_q = sum(d['quantity'])
            pcs_m = np.median(d["pcs_m"])
            p_total = t_q/self.quantity
            a_row = [code, codes[code], t_q, np.round(pcs_m, 3), np.round(p_total, 3)]
            a_table.append(a_row)

        needs_formatting = {
            "text": {
                "use": "first row", "figure_id": "make_fragmented_plastics_table"}
        }

        top_row = row_and_column_label_formatter(lang=self.lang, rows_columns=needs_formatting)
        result = [top_row["first row"], *a_table]

        return result

    def make_alternate_group_summary(self, sensors: list[Sensor] = None):
        """

        :param sensors:
        :return:
        """

        results = [x.sensor_survey_results() for x in sensors]

        pass

    def collect_k_and_n_minus_k(self, codes: [] = None) -> {}:

        t = [x.tries_fails for x in self.sensors]
        # extract all the tries and fails for each possible value of x
        # from each sensor and sum n, n-k and n for each x for each object
        # in codes
        if codes is None:
            codes = list(t[0].keys())
        results = {}

        for code in codes:
            data = np.array([j[code] for j in t])
            instances = functools.reduce(priors.add_instances_by_sequence, [list(x.values()) for x in data])
            results.update({code: instances})
        return results

    def get_priors(self, column: str = "quantity", columns: [] = priors.columns):
        df = priors.df
        adf = df[df[self.parent_group] == self.parent_group_name].copy()
        locations = adf.location.unique()
        apriori = priors.assemble_priors(df=adf, columns=columns, column=column, locations=locations, prior=(1, 1))

        code_priors = priors.make_parameters(df=adf, column="quantity")

        return apriori

    def make_parameters_from_priors(self, a_column: pd.Series = None):

        return [x for x in a_column.values if isinstance(x, int) is False]

    # def make_code_priors(df):
    #     return for code in codes:
    def ad_d_priors_to_sensors(self):

        these_priors = self.get_priors()
        p_keys = list(these_priors.keys())
        these_sensors = self.sensor_names

        wi_th_priors = {}

        for key in list(set(p_keys) & set(these_sensors)):
            d = these_priors[key]
            this_sensor = [x for x in self.sensors if x.sensor == key]
            tries_fails = this_sensor[0].tries_fails
            labels = list(d.keys())
            for label in labels:
                # get the prior data
                this_prior = self.make_parameters_from_priors(d[label])
                # these objects were aggregated to gfoam in later studies
                if label in ["G81", "G82", "G83"]:
                    label = "Gfoam"
                # these were aggregated to gfrags in later studies
                elif label in ['G79', 'G80', 'G78', 'G76', 'G75']:
                    label = "Gfrags"
                else:
                    pass
                if label in tries_fails.keys():
                    if isinstance(tries_fails[label], dict):
                        this_data = list(tries_fails[label].values())
                        # this_data = [[x[0]+1, x[1]+1] for x in ts_data]
                        print(f"{key} {label} was in dict form")
                    else:
                        print(f"\n!! {key} { label}  was good to go !!\n")
                        this_data = tries_fails[label]
                    wi_th_priors.update({label: priors.add_instances_by_sequence(this_prior, this_data)})
            this_sensor[0].tries_fails = wi_th_priors


    def sample_this(self, k, n_minus_k, beta=beta, dist=bernoulli):

        p = beta(k, n_minus_k).rvs(1)
        proceed = dist(p).rvs(1)
        assert p < 1, "This should always be less than 1"
        return proceed[0]

    def walk_a_column(self, a_col: [] = None, n_samples: int = None):
        results = []
        for _ in np.arange(n_samples):
            for i, val in enumerate(a_col):
                print(val)
                # locations that do not have priors
                # need the uniform prior to start the sequences
                # if the first value has v[1] > 0 at i = 0
                # then this is a new location and needs the uniform
                # added.
                k = val[0]
                if i == 0 and val[1] == 0:
                    # there are no samples
                    # for this item
                    break
                elif k == 0 and val[1] > 0 and i == 0:
                    k = 1
                else:
                    n_minus_k = val[1] - k
                    if n_minus_k == 0:
                        # that means there was no
                        # inititial set on nminusk
                        n_minus_k = 1
                    else:
                        pass
                    print(k, n_minus_k)
                    proceed = self.sample_this(k, n_minus_k)
                    if proceed == 0:
                        results.append(i)
                        break
                    else:
                        pass

        return results

    def make_prediction(self):
        self.ad_d_priors_to_sensors()
        samples_per_sensor = {x.sensor: x.sensor_nsamps() for x in self.sensors}
        sen_sors = {x.sensor: x.tries_fails for x in self.sensors}

        predictions = {}
        for a_sensor in self.sensor_names:
            n_samples = samples_per_sensor[a_sensor]
            params = sen_sors[a_sensor]
            keys = list(params.keys())
            for key in keys:
                d = params[key]
                if isinstance(d, dict):
                    d = list(d.values())
                predicted = self.walk_a_column(d, n_samples)
                if key in predictions.keys():
                    predictions[key] += predicted
                else:
                    predictions.update({key: predicted})
        return predictions

# 'limmat_zuerich_suterdglauserp'
def check_for_place_names(surveys):
    """Checks the geographic attributes on the proposed Survey classes against the current index of place names

    Raises a KeyError if the sensor or location name is not in the current index.
    """

    # Call the sensor details class and
    # initiate a place_names map
    d = SensorDetails()
    d.set_place_names_map()
    dmap = d.place_names_map

    for x in list(surveys.values()):
        try:
            dmap["survey_area"][x.survey_area]
        except KeyError as err:
            print("There is no key for this location in the survey areas", err)
        try:
            dmap["feature"][x.feature]
        except KeyError as err:
            print("There is no key for this location in the features", err)
        try:
            dmap["city"][x.city]
        except KeyError as err:
            print("There is no key for this location in the cities", err)

        finally:
            return None


def create_survey_db(connection: str = None):

    code_definitions = CodeDataClass()

    if connection is None:
        with open("dc_surveys.csv", "r") as a_file:
            a_reader = csv.reader(a_file, delimiter=",")
            # this data has headers
            next(a_reader)
            surveys = {}
            for row in a_reader:
                thing_args = [
                    row[2],
                    code_definitions.groupname[row[2]],
                    int(row[4]),
                    float(row[3])
                ]
                if row[5] in surveys.keys():
                    surveys[row[5]].things.append(Thing(*thing_args))
                else:
                    instance = row[5]
                    sensor = row[0]
                    survey_area = row[6]
                    feature = row[7]
                    city = row[8]
                    date = row[1]
                    surveys.update({instance: Survey(instance, sensor, city, feature, survey_area, date)})
                    surveys[instance].things.append(Thing(*thing_args))
    else:
        print(connection)
        pass

    keys = list(surveys.keys())

    assert isinstance(surveys[keys[0]].things[0].group, str), "There should be a string here"

    check_for_place_names(surveys)

    return surveys


def make_sensor_class(**kwargs):
    """Constructor for Sensor objects"""

    return Sensor(**kwargs)


def select_sensors(parent: str = None, search: str = None, child_group: str = None, surveys: list[Survey] = None):
    """Makes groups of Sensors given a parent, search and children variable"""

    s = [x for x in surveys if slugify(x.__dict__[parent]) == search]
    child_groups = list({x.__dict__[child_group] for x in s})

    results = []
    for a_name in child_groups:
        these_instances = [x for x in s if x.__dict__[child_group] == a_name]
        these_sensors = list({x.sensor for x in these_instances})

        for a_sensor in these_sensors:
            args = {
                "sensor": a_sensor,
                "member": a_name,
                "surveys": [x for x in these_instances if x.sensor == a_sensor]
            }
            # this is one sensor, or the collection of surveys defined by a sensor
            # before moving on the method sensor_tries_fails is called
            # getting this value now prevents iterating through the collection of
            # all the sensors later.
            result = Sensor(**args)
            result.sensor_tries_fails()
            # once it is called append it to the list

            results.append(result)

    return results


def create_report_class(
        report_name: str = None, parent_group_name: str = None, parent_group: str = None,
        child_group: str = None, sensors: list[Sensor] = None, lang: str = None,
        user: str = None, date: str = None):

    # here the generator is emptied.
    generated = sensors

    report_args = dict(
        report_name=report_name, child_group=child_group, parent_group_name=parent_group_name,
        sensors=generated, user=user, date=date, lang=lang, parent_group=parent_group)

    a_report = ReportMethods(**report_args)
    # initiate some attributes:
    a_report.set_and_label_children()
    a_report.set_report_number_of_sensors()
    a_report.set_report_quantity()
    a_report.set_report_date_range()
    a_report.set_report_survey_totals()
    a_report.set_report_code_totals()
    a_report.set_report_max_min()
    a_report.set_report_number_of_samples()
    a_report.set_report_quantiles()
    a_report.report_code_fail_rate()
    a_report.set_report_sensor_names()

    return a_report


def make_this_report(report_name: str = "Lac Léman", pg: str = "feature", cg: str = "sensor", pgn: str = "lac-leman", d_vals: [] = None):

    if d_vals is None:
        s = create_survey_db()
        surveys = list(s.values())
        d_vals = select_sensors(parent=pg, search=pgn, child_group=cg, surveys=surveys)


    args_s = dict(parent=pg, search=pgn, child_group=cg, surveys=d_vals)

    args = dict(report_name=report_name, parent_group_name=pgn, parent_group=pg, child_group=cg, sensors=d_vals, lang="en")
    m = create_report_class(**args)
    return m




# testing functions
def elapsed_time(last_time, inspector_items):

    this_time = time.process_time()
    elapsed = this_time - last_time
    a, b = inspector_items

    line = f"TIME to execute from {a},  {b}, elapsed time: {elapsed}"
    config.append_to_log(line)


def inspect_and_time(last_time, method_context) -> None:
    elapsed_time(last_time, method_context)
    return None


def mixer(start: int = 0, end: int = 3) -> ():
    """Collects two numbers between 0 and 3. The second number must be <= the first."""

    parent = random.randint(start, end)
    child = parent + 1

    while child > parent:
        child = random.randint(start, end)
    return parent, child


def selector(surveys: list[Survey] = None) -> ():

    mixed = mixer()

    select_group = {i: x[0] for i, x in enumerate(config.sensor_columns())}
    parent_group = select_group[mixed[0]]
    child_group = select_group[mixed[1]]
    # get the pool of possible selections
    # instantiate the map
    if parent_group == "location":
        parent_group = "sensor"
    if child_group == 'location':
        child_group = "sensor"
    places = [x.__dict__[parent_group] for x in surveys]
    # this is all the current survey or data identifiers for the group
    # selected by mixed[0]
    pool = np.unique(places)
    end = len(pool)-1
    # select a specific location from the pool
    p, _ = mixer(0, end)
    # therefore the subject of the report is
    # or the search term is
    parent = pool[p]

    choose_language = random.randint(0, 1)
    languages = ["en", "de"]
    lang = languages[choose_language]

    # this is a generator:
    some_sensors = select_sensors(
        parent=parent_group,
        search=parent,
        surveys=surveys
    )

    args = dict(
        report_name="Random",
        parent_group_name=parent,
        parent_group=parent_group,
        child_group=child_group,
        sensors=some_sensors,
        lang=lang,
        user="Random tester")

    a_new_rpt = create_report_class(**args)

    return a_new_rpt


def call_report_functions(a_new_rpt):
    t = time.process_time()
    a_new_rpt.set_report_code_group_stats()
    inspect_and_time(t, ["set_report_code_group_stats", "test"])
    a_new_rpt.set_report_most_common_codes()
    inspect_and_time(t, ["set_report_most_common_codes", "test"])
    a_new_rpt.make_report_scatter_chart()
    inspect_and_time(t, [".make_report_scatter_chart", "test"])
    a_new_rpt.set_report_survey_totals()
    inspect_and_time(t, ["set_report_survey_totals", "test"])
    a_new_rpt.make_cumulative_distribution_chart()
    inspect_and_time(t, ["make_cumulative_distribution_chart", "test"])
    a_new_rpt.make_report_code_summary()
    inspect_and_time(t, ["make_report_code_summary", "test"])
    a_new_rpt.make_heat_map_codes()
    inspect_and_time(t, ["make_heat_map_codes", "test"])
    a_new_rpt.make_table_survey_total_summary()
    inspect_and_time(t, ["make_table_survey_total_summary", "test"])
    a_new_rpt.make_fragmented_plastics_table()
    inspect_and_time(t, ["make_fragmented_plastics_table", "test"])
    a_new_rpt.make_summary_table_of_child_activities()
    inspect_and_time(t, ["make_summary_table_of_child_activities", "test"])

    return a_new_rpt


def load_a_test(choice: str = None, survey_db: list[Survey] = None):

    if survey_db is None:
        survey_db = create_survey_db()

    vals = list(survey_db.values())
    if choice == "a":
        surveys = [x for x in vals if x.sensor == "maladaire"]
        kwargs = {
            "survey_area": "rhone",
            "feature": "lac-leman",
            "city": "la-tour-de-peilz",
            "member": "maladaire",
            "sensor": "maladaire",
            "surveys": surveys
        }

        maladaire = make_sensor_class(**kwargs)
        args = dict(
            report_name="Option a",
            parent_group_name="maladaire",
            parent_group="sensor",
            child_group="sensor",
            sensors=[maladaire],
            lang="de",
            user="tester")
        a_new_rpt = create_report_class(**args)
        call_report_functions(a_new_rpt)
        return a_new_rpt

    if choice == "b":
        some_sensors = select_sensors(
            parent="feature",
            search="lac-leman",
            surveys=vals
        )

        args = dict(
            report_name="Option b",
            parent_group_name="lac-leman",
            parent_group="feature",
            child_group="city",
            sensors=some_sensors,
            lang="de",
            user="tester")
        a_new_rpt = create_report_class(**args)
        # call_report_functions(a_new_rpt)
        return a_new_rpt
    elif choice == "c":
        some_sensors = select_sensors(
            parent="survey_area",
            search="aare",
            surveys=vals
        )

        args = dict(
            report_name="Option c",
            parent_group_name="aare",
            parent_group="survey_area",
            child_group="city",
            sensors=some_sensors,
            lang="de",
            user="tester")

        a_new_rpt = create_report_class(**args)
        # call_report_functions(a_new_rpt)
        return a_new_rpt
    elif choice == "d":

        some_sensors = select_sensors(
            parent="city",
            search="Zürich",
            surveys=vals
        )

        args = dict(
            report_name="Option d",
            parent_group_name="Zürich",
            parent_group="city",
            child_group="sensor",
            sensors=some_sensors,
            lang="de",
            user="tester")

        a_new_rpt = create_report_class(**args)
        # call_report_functions(a_new_rpt)
        return a_new_rpt
    elif choice == 'r':

        a_new_rpt = selector(vals)
        # call_report_functions(a_new_rpt)
        return a_new_rpt

    elif choice == 'r+':
        j = 0
        for i in np.arange(30):
            while j < 30:

                a_new_rpt = selector(vals)
                # call_report_functions(a_new_rpt)
                j += 1
                print(f"!!!! THIS IS I {i}, THIS J {j}")

    return a_new_rpt


def one_plus_two(one: int = None, two: int = None):
    return one + two







