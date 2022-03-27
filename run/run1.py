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
a1 = Domain('less', AlterLinearMF(30, 1.0, 70, 0))
a2 = Domain('more', AlterLinearMF(40, 0, 80, 1.0))

h1 = Domain('good', SigmoidMF(25, 0.3))
h2 = Domain('poor', SigmoidMF(25, -0.7))

# Define fuzzy variables for both input and output fuzzy variables.
# If we use Sugeno method, we need to define output fuzzy variable separately.
alc: FuzzyVariable = FuzzyVariable('alc', 0, 1, a1, a2)
health: FuzzyVariable = FuzzyVariable('health', 0, 1, h1, h2)

# Define membership functions for Sugeno method.
# If we have dataset, we call 'comp_linear_reg_coff' function to compute coefficients and intercepts.
# Pass

# Define a output fuzzy variable for Sugeno method.
# Pass

# Define boundary values
alc_x_min = 0
alc_x_max = 100
alc_x_resolution = (alc_x_max - alc_x_min) * 5
health_x_min = 0
health_x_max = 50
health_x_resolution = (health_x_max - health_x_min) * 5

# Check for validation
if not check_valid([alc_x_min, alc_x_max], [health_x_min, health_x_max]):
    raise Exception('Invalid Input Values')

# Define rules
# Note: if we use Sugeno defuzzification method, there can not be more than two assertions in conditions.
rule = Rule()
conditions1, conclusion1 = rule.parse_rule('if (alc is less) then (health is good)')
conditions2, conclusion2 = rule.parse_rule('if (alc is more) then (health is poor)')

# Initial fuzzy engine
fuzzy_system = FuzzyEngine([alc], health, FuzzyRuleOperatorType.AND, health_x_min, health_x_max)

# Add fuzzy rules
fuzzy_system.rules.append([conditions1, conclusion1])
fuzzy_system.rules.append([conditions2, conclusion2])

# Fuzzy inference
# For Mamdani and Tsukamoto methods, we call 'calculate' function.
# For Sugeno method, we call 'calculate_sugeno' function.
fuzzy_matrix, conditions, conclusions, fuzzy_result = fuzzy_system.calculate(OrderedDict({alc: 65.0}))

# Defuzzification
# Note:
#   (1) If we set a SugenoFuzzyVariable as our output variable, then we can only defuzzify with Sugeno defuzzification.
#   (2) If we set a FuzzyVariable as our output variable, then we can always use Mamdani defuzzification.
#   (3) If we set a FuzzyVariable as our output variable, and the membership functions of the output variable is strictly increasing or decreasing, then we can use both Mamdani and Tsukamoto defuzzification.
defuzzifier_m = Mamdani(health_x_min, health_x_max)
d_m = defuzzifier_m.get_value(fuzzy_result)
s_m = "Anticipated health index (Mamdani): " + str(d_m)
print(s_m)

defuzzifier_t = Tsukamoto([InverseSigmoidMF(25, 0.3), InverseSigmoidMF(25, -0.7)])
d_t = defuzzifier_t.get_value(conclusions, health_x_min, health_x_max)
s_t = 'Anticipated health index (Tsukamoto):' + str(d_t)
print(s_t)

# Print results to txt file
write_text(config, s_m, s_t)

# Plot membership function with input values
plot_mf(alc, 65.0, alc_x_min, alc_x_max, alc_x_resolution, config)

# Plot individual fuzzy rules
# We only call this function for Mamdani and Tsukamoto defuzzification.
plot_conclusions(health, health_x_min, health_x_max, health_x_resolution, conclusions, config)

# Plot the aggregated fuzzy result rule
# We only call this function for Mamdani and Tsukamoto defuzzification.
plot_fuzzy_result(health_x_min, health_x_max, health_x_resolution, fuzzy_result, config)

# Plot the fuzzy result rule with the defuzzified centroid
# We only call this function for Mamdani and Tsukamoto defuzzification.
plot_fuzzy_result_with_center(health_x_min, health_x_max, health_x_resolution, fuzzy_result, {'Mamdani': d_m, 'Tsukamoto': d_t}, config)


if __name__ == '__main__':
    pass
