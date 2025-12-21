WITH source AS (
    SELECT * FROM {{ source('olist_raw', 'raw_geolocation') }}
)

SELECT
    CAST(NULLIF(JSON_EXTRACT_SCALAR(data, '$.geolocation_zip_code_prefix'), '') AS INT64) AS geolocation_zip_code_prefix,
    CAST(NULLIF(JSON_EXTRACT_SCALAR(data, '$.geolocation_lat'), '') AS FLOAT64) AS geolocation_lat,
    CAST(NULLIF(JSON_EXTRACT_SCALAR(data, '$.geolocation_lng'), '') AS FLOAT64) AS geolocation_lng,
    JSON_EXTRACT_SCALAR(data, '$.geolocation_city') AS geolocation_city,
    JSON_EXTRACT_SCALAR(data, '$.geolocation_state') AS geolocation_state
FROM source