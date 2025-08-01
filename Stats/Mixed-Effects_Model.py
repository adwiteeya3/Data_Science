import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Sample Hierarchical Data
np.random.seed(42)
n_schools = 10
n_students_per_school = 20
school_ids = np.repeat(np.arange(n_schools), n_students_per_school)

# Simulate school-level effects
school_effects = np.random.normal(0, 2, n_schools)[school_ids]

# Simulate student-level features
study_hours = np.random.uniform(1, 10, n_schools * n_students_per_school)
iq = np.random.normal(100, 15, n_schools * n_students_per_school)

# Simulate grades with fixed effects and random school effects
grades = 50 + 2 * study_hours + 0.5 * iq + school_effects + np.random.normal(0, 5, len(school_ids))

data = pd.DataFrame({
    'school_id': school_ids,
    'study_hours': study_hours,
    'iq': iq,
    'grades': grades
})

# Fit a Mixed Linear Model
# 'grades ~ study_hours + iq' are fixed effects
# 'C(school_id)' is for the random intercept for each school
model = smf.mixedlm("grades ~ study_hours + iq", data, groups=data["school_id"])
result = model.fit()

print("\nHierarchical/Mixed-Effects Model Summary:")
print(result.summary())
