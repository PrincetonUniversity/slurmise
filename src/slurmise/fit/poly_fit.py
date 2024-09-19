import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass, asdict

# Generate a polynomial fit for the runtime data using sklearn
from sklearn.externals import joblib as sk_joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from .resource_fit import ResourceFit
from ..job_data import JobData


@dataclass
class PolynomialFit(ResourceFit):
    model: Optional[Pipeline] = None
    degree: int

    def save(self):
        super().save()
        if self.model is not None:
            modelpath = self.path / "model.pkl"
            sk_joblib.dump(self.model, str(modelpath))

    def fit(self, jobs: list[JobData], **kargs):

        df = pd.json_normalize([asdict(job) for job in jobs])

        # Convert categorical columns to category type
        for col in df.columns:
            if col.startswith("categorical."):
                df[col] = df[col].astype("category")

        # Rename the categorical columns, drop .categorical prefix
        df.columns = [col.replace("categorical.", "") for col in df.columns]

        # Do the same for numerical columns
        df.columns = [col.replace("numerical.", "") for col in df.columns]

        # Get the numerical columns
        X = df.drop(columns=["job_name", "slurm_id", "memory", "runtime"])

        # Transform features
        categorical_features = [
            name for name in X.columns if X[name].dtype == "category"
        ]
        numerical_features = [
            name for name in X.columns if name not in categorical_features
        ]

        categorical_transformer = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist")),
            ]
        )
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=np.max)),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        # We are doing polynomial regression, so we need to add polynomial features
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        model = LinearRegression()

        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("poly", poly), ("model", model)]
        )

        self.model = pipeline.fit(X)
