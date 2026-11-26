# Models

Exported `.tflite` files are placed here and served by `GET /model/latest` (newest file wins). They are not committed to Git.

Files are named `67counter_v<N>.tflite`, see the model contract in `docs/data_format.md`. Member D produces them and Member C copies each approved one here.
