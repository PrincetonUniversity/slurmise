from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .job_data import JobData


def jobs_to_pandas(jobs: list[JobData]):
    """
    Convert a list of JobData objects to a pandas DataFrame. The DataFrame will have
    columns for each category and numeric feature, and will not include the job_name,
    slurm_id, or cmd fields since they are not features.

    :param jobs: A list of JobData objects
    :type jobs: list[JobData]
    :return: A pandas DataFrame with columns for each category and numeric feature
    :rtype: pd.DataFrame

    """

    df = pd.json_normalize([asdict(job) for job in jobs])

    # Convert category columns to category type
    for col in df.columns:
        if col.startswith("categories."):
            df[col] = df[col].astype("category")

    # Rename the categories, drop .category prefix
    df.columns = [col.replace("categories.", "") for col in df.columns]

    # If a column is numeric, but it's dtype is object, it may be a numpy array
    for col in df.columns:
        new_col_name = col.replace("numerics.", "")

        if col.startswith("numerics.") and df[col].dtype == "object":
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
                    msg = f"Numerics feature {new_col_name} is a numpy array of different sizes. "
                    msg += "Numpy arrays are supported only if they are all the same size."
                    raise ValueError(msg)

            else:
                msg = "Numerics columns must be scalars or equal length numpy arrays"
                raise ValueError(msg)

    df.columns = [col.replace("numerics.", "") for col in df.columns]

    # Drop job_name and slurm_id columns since they are not features
    cols_to_drop = {"job_name", "slurm_id", "cmd"}.intersection(df.columns)
    df = df.drop(columns=cols_to_drop)

    # Sort columns to ensure consistent ordering across platforms
    df = df[sorted(df.columns)]

    # Transform features
    categories = sorted([name for name in df.columns if df[name].dtype == "category"])
    numerics = sorted([name for name in df.columns if name not in categories and name not in ["memory", "runtime"]])

    return df, categories, numerics
