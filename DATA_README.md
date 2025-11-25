# Data Files

This project requires large data files that are not included in the Git repository.

## Required Files (Not in Repository)

The following large files are excluded from version control:

- `bigdata_preprocess.csv` (179.34 MB) - Preprocessed electricity consumption data
- `kafka_producer/consumed_messages.json` (1888.89 MB) - Kafka consumer message logs
- `kafka_producer/consumed_messages_2.json` (811.16 MB) - Additional Kafka message logs

## Setup Instructions

1. Obtain or generate the data files
2. Place them in the following locations:
   - `/opt/electricity-pipeline/bigdata_preprocess.csv`
   - `/opt/electricity-pipeline/kafka_producer/consumed_messages.json`
   - `/opt/electricity-pipeline/kafka_producer/consumed_messages_2.json`

## How to Generate Data

[Add instructions on how to generate these files, e.g.:]
- Run the data preprocessing pipeline: `python scripts/preprocess.py`
- Or contact the data administrator for access
