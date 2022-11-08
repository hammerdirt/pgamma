# -*- coding: utf-8 -*-
"""
Make changes here

"""

import numpy as np
import csv
import json
import datetime as dt
from collections import Counter


fail_rate = 0.5
quantiles = [0.5, .25, .5, .75, .95]
language = "en"
alt_rate = 1
weight_denom = 1000
fails = np.arange(start=0, stop=101, step=1)
rate_label = {1: "pcs/m", 100: "pcs/100m"}
rate = 100

# FILES and DATASOURCES

current_log_file = "current_log_file.txt"
frag_path = "data/frag_plastic_foams.csv"

# COLUMNS AND DATATYPES
dims_data_columns = [
            'mac_plast_w',
            'mic_plas_w',
            'total_w',
            'length',
            'area',
            'times'
        ]


def sensor_columns(sensor_defs: list[()] = None) -> list[()]:
    """Sets the data-types of the sensor definitions

    :param sensor_defs: A list of tuples ("column-name", "datatype")
    :return:If no alternate is provided the data-types for the default package are returned
    """
    if not sensor_defs:
        sensor_defs = [
            ('location', 'U50'),
            ('sensor', 'U50'),
            ('city', 'U50'),
            ('feature', 'U50'),
            ('survey_area', 'U50'),
            ('% to buildings', 'f8'),
            ('% to woods', 'f8'),
            ('intersects', 'f8')
        ]

    return sensor_defs

def code_data(code_defs: list[()] = None) -> list[()]:
    """Sets the data-types of the object codes data

    The objects collected are grouped according to a code value that describes the object type,
    defines the possible material and use.

    :param code_defs: A list of tuples ("column-name", "datatype")
    :type code_defs: list[(str, str), ... (str, str)]
    :return:If no alternate is provided the data-types for the default package are returned
    """

    if not code_defs:
        code_defs = [('code', 'U10'),
                     ('material', 'U20'),
                     ('description', 'U500'),
                     ('groupname', 'U50'),
                     ('parent_code', 'U15')]

    return code_defs


# TESTS AND LOGS
def start_log(file_name: str = None):

    if file_name is None:
        file_name = current_log_file

    today = dt.datetime.today()
    with open(file_name, "w") as file:
        file.write(f"The current log file: {today}\n")
    print("log started")


def append_to_log(file_name: str = None, line: str = "This line should have a real message") -> None:

    if file_name is None:
        file_name = current_log_file

    with open(file_name, "a") as a_file:
        a_file.write(f"{line}\n")



# FORMATTERS


def date_format(lang: str = None):

    if lang is None:
        lang = language
    formats = {
        "en": "%Y-%m-%d"
    }
    return formats[lang]


def dictionary_keys_for_fails(trys: [] = fails):
    """The fail rate is based on the equality 'X > i' where X is the quantity found af an object at one survey and i
    is a number between 0 and <fails> default is 50. The dictionary keys represent the condition 'x>i' """

    # success is when the quantity found is 0
    # which means there are two possible solutions
    # x = 0 and x > 0.
    labels = [i for i in trys]

    return labels

text_maps_for_standard_figures = {
    "make_table_survey_total_summary": {
        "de": [
            "Berichtsname",
            "Datumsbereich",
            "N Proben",
            "N Sensoren",
            "N Stücke",
            f"Median {rate_label[alt_rate]}",
            f"Minimum {rate_label[alt_rate]}",
            f"Maximale{rate_label[alt_rate]}"
        ],
        "fr": [
            "Nom du rapport",
            "Plage de dates",
            "N échantillons",
            "N capteurs",
            "N pièces",
            f"Médian {rate_label[alt_rate]}",
            f"Minimum {rate_label[alt_rate]}",
            f"Maximum {rate_label[alt_rate]}"
        ]
    },
    "make_summary_table_of_child_activities": {
        "fr": [
            "Site",
            "N échantillons",
            "N pièces",
            f"Médiane {rate_label[alt_rate]}",
            "Poids du plastique",
            "Poids total",
            "N mètres",
            "N mètres²",
            "N minutes"
        ],
        "de": [
            "Merkmal",
            "N Proben",
            "N Stücke",
            f"Median {rate_label[alt_rate]}",
            "Kunststoffgewicht",
            "Gesamtgewicht",
            "N Meter",
            "N Meter²",
            "N Minuten"
        ]
    },
    "make_report_code_summary": {
        "en": [
            "Code",
            "Description",
            "N pieces",
            rate_label[alt_rate],
            "% of total",
            "Fail rate"
        ],
        "fr": [
            "Code",
            "Déscription",
            "Pièces",
            rate_label[alt_rate],
            "% du total",
            "Taux d'échec"
        ],
        "de": [
            "Code",
            "Beschreibung",
            "Gesamt",
            rate_label[alt_rate],
            "% der Gesamtmenge",
            "Ausfallrate"
        ]
    },
    "make_report_scatter_chart-readme": {
        "en": "Survey totals for all locations",
        "de": "Gesamtergebnisse der Umfrage für alle Standorte",
        "fr": "Totaux des enquêtes pour tous les sites"
    },
    "make_fragmented_plastics_table": {
        "en": ["Code", "Description", "N pieces", rate_label[alt_rate], "% of total"],
        "de": ["Code", "Beschreibung", "Gesamt", rate_label[alt_rate], "% der Gesamtmenge"],
        "fr": ["Code", "Déscription", "Total", rate_label[alt_rate], "% du total"]
    },
    "make_cumulative_distribution_chart": {
        "en": "Ratio of samples",
        "de": "Verhältnis der Proben",
        "fr": "le proportion d'échantillons",
    },
}

# UTILITIES


def get_csv_data(source: str = None, dtype: list[()] = None, headers: bool = True) -> np.ndarray:

    with open(source, "r") as f:
        some_rows = csv.reader(f)
        if headers:
            next(some_rows)
        else:
            pass
        survey_data = np.array([tuple(row) for row in some_rows], dtype=dtype)

    return survey_data


def get_json_data(source: str = None) -> dict:

    with open(source, "r") as a_file:
        data = json.load(a_file)

    return data


def retrieve_data(
        source: str = None, dtype: list[()] = None, ext: str = None, ext_parameters: dict = None) -> []:
    if ext == "csv":
        data = get_csv_data(source=source, dtype=dtype)
        return data
    if ext == "json":
        data = get_json_data(source=source)
        return data
    elif ext == "db":
        # make db connection with ext_parameters
        pass
    elif ext == "www":
        # make http connection with ext_parameters
        pass
    else:
        raise ValueError


def call_a_class_method(a_thing: object = None, a_method_name: str = None, is_callable: bool = True) -> ():
    """Calls the requested method on a_thing, a_method name must correspond to either a method or attribute of a_thing.

    The flag is_callable differentiates between a class method and an existing attribute. The called method and the
    is_callable flag must correspond. A value error is raised if a_method_name corresponds to an attribute
    (is_callable=False) and the flag is set to true, for example.

    If the value of the called method is None a message is returned

    :param a_thing:
    :param a_method_name:
    :param is_callable:
    :return: Tuple of the values from the executed method.
    """

    # check the methods and attributes in a_thing
    # thnx stackoverflow https://stackoverflow.com/questions/1911281/how-do-i-get-list-of-methods-in-a-python-class
    method_list = [func for func in dir(a_thing) if func[:2] != '__']
    attributes = a_thing.__dict__.keys()
    # if the value of result changes it will be returned to the requestor
    result = None
    try:
        if a_method_name in attributes and not is_callable:
            result = getattr(a_thing, a_method_name)
        elif a_method_name in method_list and is_callable:
            x = getattr(a_thing, a_method_name)
            result = x()
    except ValueError as err:
        print("The method name and the is_callable flag need to correspond.")
    if result is not None:
        return result
    else:
        # Or a message is returned
        print('{"message": "Those arguments raised no errors but produced no values"}')
        raise ValueError



def collect_methods(things: [] = None, a_key: str = None, a_func: callable = None,  **kwargs):
    """Collects class methods and groups by the values in keys

    If keys is None the default is to use the code field as a key
    """

    if not a_key:
        a_key = "code"

    keys = {x.__dict__[a_key] for x in things}
    result = {}
    for key in keys:
        x = [call_a_class_method(x, **kwargs) for x in things if x.__dict__[a_key] == key]
        operated = a_func(x)
        result.update({key:operated})
    assert result is not None, "The dict should have something in there"
    return result

# def choose_method(a_param: str = None, sensor: any = None) -> str:
#     available_methods = {
#
#         "pcs_m": sensor.sensor_code_results_pcs_m,
#         "% of total": sensor.sensor_code_results_qty,
#         "group_pcs_m": sensor.sensor_code_group_pcs_m,
#         "% of total group": sensor.sensor_code_group_totals
#     }
#     this_method = available_methods[a_param]
#     return this_method
#
# def retrieve_values_from_sensors(
#         sensors: [] = None, method_name: str = None, key_attribute: str = None,
#         key_value: str = None, a_param: str = None) -> []:
#
#     print("\n\n Retrieving \n\n")
#     for x in sensors:
#         print(x.__dict__[key_attribute])
#         print(key_value)
#         print(choose_method(a_param, x)())
#     these_values = [choose_method(a_param, x) for x in sensors if x.__dict__[key_attribute] == key_value]
#     print(these_values[0].keys())
#     return these_values


def get_dimensional_data(data: [] = None, dim_data: dict = None) -> ():
    """Returns the dimensional and participation data for each survey"""

    # if the dimensional data is not provided use the defaults
    if dim_data is None:
        dim_data = retrieve_data(source="location_cumulative_statistics.json", ext="json")

    # collect the sensor id for each sensor
    sensors = [x.sensor for x in data]

    assert len(dim_data) >= len(sensors), "All of sensors should be in dim_data"
    not_in_dims_data = [x for x in sensors if x not in dim_data.keys()]

    assert len(not_in_dims_data) == 0, "This should be an empty list"

    # sum the results for each sensor
    res = [Counter(dim_data[x]) for x in sensors]
    results = sum(res, Counter())
    weights = (results['mac_plast_w'] + results['mic_plas_w'], results["total_w"])
    measures = (results["length"], results["area"])
    times = results["times"]

    return weights, measures, times

