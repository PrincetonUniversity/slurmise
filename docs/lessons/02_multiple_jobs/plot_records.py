"""Plot memory against intensity for the perfectScaler records in slurmise.h5, saved to records.png."""

from matplotlib.figure import Figure

from slurmise.job_data import JobData
from slurmise.job_database import JobDatabase

with JobDatabase.get_database("slurmise.h5") as db:
    jobs = db.query(JobData(job_name="perfectScaler"))

fig = Figure(figsize=(5, 3.5), layout="constrained")
ax = fig.subplots()
ax.scatter([job.numerics["intensity"] for job in jobs], [job.memory for job in jobs])
ax.set(xlabel="--intensity", ylabel="memory used (MB)", title=f"{len(jobs)} perfectScaler records")

fig.savefig("records.png", dpi=100)
print(f"Plotted {len(jobs)} records to records.png")
