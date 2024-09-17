from slurmise.job_data import JobData

import numpy as np

rng = np.random.default_rng(42)

NUM_JOBS = 20
jobs = []
for i in range(NUM_JOBS):
    
    if i % 2 == 0:
        mode = "1"
        factor = 1
    else:
        mode = "2"
        factor = 1.1
    
    N = ((i//2)+1)*100
    
    # M effects performance very slightly
    M = (i//2)+1 * 10
    
    # Runtime is quadratic in N, with some noise
    runtime = factor*(N**2 + rng.normal(0, 0.1*N**2)) + M*rng.normal(0, 0.1)
    memory = N*10 + rng.normal(0, 0.1)

    job = JobData(job_name=f"fake_job", slurm_id=f"{i}", categorical={"--mode": mode}, numerical={"-N": N, "-M": M}, memory=memory, runtime=runtime)

    jobs.append(job)

# for job in jobs:
#     print(job)

# Now build a pandas DataFrame from the list of JobData dataclass objects
import pandas as pd
from dataclasses import asdict    
df = pd.json_normalize([asdict(job) for job in jobs])

# Convert categorical columns to category type
for col in df.columns:
    if col.startswith("categorical."):
        df[col] = df[col].astype("category")

# Rename the categorical columns, drop .categorical prefix
df.columns = [col.replace("categorical.", "") for col in df.columns]

# Do the same for numerical columns
df.columns = [col.replace("numerical.", "") for col in df.columns]

print(df)
print(df.dtypes)

# Generate a polynomial fit for the runtime data using sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Get the numerical columns
X = df.drop(columns=["job_name", "slurm_id", "memory", "runtime"])

# Transform features
categorical_features = [name for name in X.columns if X[name].dtype == "category"]
numerical_features = [name for name in X.columns if name not in categorical_features]

print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist")),
    ]
)
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy=np.max)), ("scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# We are doing polynomial regression, so we need to add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
model = LinearRegression()

pipeline = Pipeline([("preprocessor", preprocessor), ("poly", poly), ("model", model)])

