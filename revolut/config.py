import os
from pathlib import Path

#Set path
schema_path = Path(
        os.environ['HOME'],
        'Codes',
        'Challenge - Data Scientist - Product',
        'misc',
        'schemas.yaml'
    )
data_path =  Path(
        os.environ['HOME'],
        'Codes',
        'Challenge - Data Scientist - Product',
        'data'
    )

#Set db variables
host='localhost'
port=54320
dbname='ht_db'
user='postgres'
