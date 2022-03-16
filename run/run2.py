import os
import sys
sys.path.append(os.path.abspath('../src'))

import yaml
import numpy as np
from collections import OrderedDict

from domain import *
from variable import *
from mf import *
from rule import *
from fuzzy_engine import *
from defuzzifier import *
from inv_mf import *
from utils import *


# Load configuration
with open('../config.yml', 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Create output folder
abs_dir = os.path.abspath('../output')
file_name = os.path.splitext(__file__)[0]
output_dir = os.path.join(abs_dir, file_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
config['output_path'] = output_dir

# Define membership functions for both input and output fuzzy variables.
# If we use Sugeno method, we need to define membership functions separately.
s1 = Domain('less', SigmoidMF(11, -2))
s2 = Domain('more', SigmoidMF(12, 1))

m1 = Domain('light', SigmoidMF(10, -1))
m2 = Domain('heavy', SigmoidMF(10, 1))

iq1 = Domain('poor', AlterLinearMF(30, 1.0, 80, 0))
iq2 = Domain('avg', GaussianMF(50, 7))
iq3 = Domain('good', AlterLinearMF(40, 0, 70, 1.0))

# Define fuzzy variables for both input and output fuzzy variables.
# If we use Sugeno method, we need to define output fuzzy variable separately.
sport: FuzzyVariable = FuzzyVariable('sport', 0, 1, s1, s2)
media: FuzzyVariable = FuzzyVariable('media', 0, 1, m1, m2)
iq: FuzzyVariable = FuzzyVariable('iq', 0, 1, iq1, iq2, iq3)

# Define membership functions for Sugeno method.
# Pass

# Define a output fuzzy variable for Sugeno method.
# Pass

# Define boundary values
sport_x_min = 0
sport_x_max = 20
sport_x_resolution = (sport_x_max - sport_x_min) * 5
media_x_min = 0
media_x_max = 20
media_x_resolution = (media_x_max - media_x_min) * 5
iq_x_min = 0
iq_x_max = 100
iq_x_resolution = (iq_x_max - iq_x_min) * 5

# Check for validation
if not check_valid([sport_x_min, sport_x_max], [media_x_min, media_x_max], [iq_x_min, iq_x_max]):
    raise Exception('Invalid Input Values')

# Define rules
# Note: if we use Sugeno defuzzification method, there can not be more than two assertions in conditions.
rule = Rule()
conditions1, conclusion1 = rule.parse_rule('if (sport is more) and (media is light) then (iq is good)')
conditions2, conclusion2 = rule.parse_rule('if ((sport is more) and (media is heavy)) or ((sport is less) and (media is light)) then (iq is avg)')
conditions3, conclusion3 = rule.parse_rule('if (sport is less) and (media is heavy) then (iq is poor)')

# Initial fuzzy engine
fuzzy_system = FuzzyEngine([sport, media], iq, FuzzyRuleOperatorType.AND, iq_x_min, iq_x_max)

# Add fuzzy rules
fuzzy_system.rules.append([conditions1, conclusion1])
fuzzy_system.rules.append([conditions2, conclusion2])
fuzzy_system.rules.append([conditions3, conclusion3])

# Fuzzy inference
# For Mamdani and Tsukamoto methods, we call 'calculate' function.
# For Sugeno method, we call 'calculate_sugeno' function.
fuzzy_matrix, conditions, conclusions, fuzzy_result = fuzzy_system.calculate(OrderedDict({sport: 10, media: 3}))

# Defuzzification
# Note:
#   (1) If we set a SugenoFuzzyVariable as our output variable, then we can only defuzzify with Sugeno defuzzification.
#   (2) If we set a FuzzyVariable as our output variable, then we can always use Mamdani defuzzification.
#   (3) If we set a FuzzyVariable as our output variable, and the membership functions of the output variable is strictly increasing or decreasing, then we can use both Mamdani and Tsukamoto defuzzification.
defuzzifier_m = Mamdani(iq_x_min, iq_x_max)
d_m = defuzzifier_m.get_value(fuzzy_result)
s_m = "Anticipated health index (Mamdani): " + str(d_m)
print(s_m)

# Print results to txt file
write_text(config, s_m)

# Plot membership function with input values
plot_mf(sport, 10, sport_x_min, sport_x_max, sport_x_resolution, config)
plot_mf(media, 3, media_x_min, media_x_max, media_x_resolution, config)

# Plot individual fuzzy rules
# We only call this function for Mamdani and Tsukamoto defuzzification.
plot_conclusions(iq, iq_x_min, iq_x_max, iq_x_resolution, conclusions, config)

# Plot the aggregated fuzzy result rule
# We only call this function for Mamdani and Tsukamoto defuzzification.
plot_fuzzy_result(iq_x_min, iq_x_max, iq_x_resolution, fuzzy_result, config)

# Plot the fuzzy result rule with the defuzzified centroid
# We only call this function for Mamdani and Tsukamoto defuzzification.
plot_fuzzy_result_with_center(iq_x_min, iq_x_max, iq_x_resolution, fuzzy_result, {'Mamdani': d_m}, config)


if __name__ == '__main__':
    pass
