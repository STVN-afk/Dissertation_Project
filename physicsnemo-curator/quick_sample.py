from datetime import datetime, timedelta

from physicsnemo_curator.domains.da.filters.stats import DataArrayStatsFilter
from physicsnemo_curator.domains.da.sinks.zarr_writer import ZarrSink
from physicsnemo_curator.domains.da.sources.era5 import ERA5Source
from physicsnemo_curator.run import run_pipeline

# Hourly timestamps for one day
times = [datetime(2020, 1, 1) + timedelta(hours=h) for h in range(24)]

# Source → Filter → Sink
pipeline = (
    ERA5Source(times=times, variables=["u10m", "v10m", "t2m"], backend="arco")
    .filter(DataArrayStatsFilter(output="output/stats.zarr", dims=("time",)))
    .write(ZarrSink(output_path="output/dataset.zarr"))
)

# Execute in parallel
results = run_pipeline(pipeline, n_jobs=4, backend="process_pool")