GRANT ALL PRIVILEGES ON DATABASE ship_ai_boa TO boa_user;

\c ship_ai_boa;

CREATE TABLE boa_entries (
   id serial PRIMARY KEY,
   task_id VARCHAR(255) NOT NULL,
   orthanc_timestamp TIMESTAMP,
   start_timestamp TIMESTAMP,
   end_timestamp TIMESTAMP,
   study_description VARCHAR(255),
   accession_number VARCHAR(255),
   series_description VARCHAR(255),
   git_hash VARCHAR(255),
   version VARCHAR(255),
   download_time FLOAT,
   inference_time FLOAT,
   num_voxels INT,
   num_slices INT,
   num_slices_resampled INT,
   bca_metrics_time FLOAT,
   totalsegmentator_metrics_time FLOAT,
   iv_contrast_phase INT,
   git_contrast INT,
   bca_regions INT,
   excel_time FLOAT,
   total_time FLOAT,
   save_persistent_time FLOAT,
   computed BOOLEAN,
   UNIQUE (task_id)
);

GRANT SELECT ON boa_entries to ship_ai_boa_public_reader;
GRANT INSERT,UPDATE ON boa_entries to ship_ai_boa_public_writer_user;
GRANT USAGE,SELECT ON SEQUENCE boa_entries_id_seq to ship_ai_boa_public_writer_user;
