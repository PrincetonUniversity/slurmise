from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .job_data import JobData


def jobs_to_pandas(jobs: list[JobData]):
    """
    Convert a list of JobData objects to a pandas DataFrame. The DataFrame will have
    columns for each categorical and numerical feature, and will not include the job_name,
    slurm_id, memory, or runtime columns.

    :param jobs: A list of JobData objects
    :type jobs: list[JobData]
    :return: A pandas DataFrame with columns for each categorical and numerical feature
    :rtype: pd.DataFrame

    """

    df = pd.json_normalize([asdict(job) for job in jobs])

    # Convert categorical columns to category type
    for col in df.columns:
        if col.startswith("categorical."):
            df[col] = df[col].astype("category")

    # Rename the categorical columns, drop .categorical prefix
    df.columns = [col.replace("categorical.", "") for col in df.columns]

    # If a column is numerical, but it's dtype is object, it may be a numpy array
    for col in df.columns:
        new_col_name = col.replace("numerical.", "")

        if col.startswith("numerical.") and df[col].dtype == "object":
            # Check if they are all numpy arrays
            if all(isinstance(row, np.ndarray) for row in df[col]):
                # Check if the column is a numpy array of all the same size
                unique_sizes = {row.shape for row in df[col]}

                if len(unique_sizes) == 1:
                    # If all the same size, expand each element of the numpy array into a new column
                    col_df = df[col].apply(pd.Series)
                    col_df.columns = [f"{new_col_name}_{i}" for i in range(col_df.shape[1])]

                    # Drop the original column and add the new columns
                    df = df.drop(columns=[col])
                    df = pd.concat([df, col_df], axis=1)

                else:
                    msg = f"Numerical feature {new_col_name} is an a numpy array of different sizes. "
                    msg += "Numpy arrays are supported only if they are all the same size."
                    raise ValueError(msg)

            else:
                msg = "Numerical columns must be scalars or equal length numpy arrays"
                raise ValueError(msg)

    df.columns = [col.replace("numerical.", "") for col in df.columns]

    # Get the numerical columns
    df = df.drop(columns=["job_name", "slurm_id", "cmd"])

    # Transform features
    categorical_features = [name for name in df.columns if df[name].dtype == "category"]
    numerical_features = [
        name for name in df.columns if name not in categorical_features and name not in ["memory", "runtime"]
    ]

    return df, categorical_features, numerical_features
