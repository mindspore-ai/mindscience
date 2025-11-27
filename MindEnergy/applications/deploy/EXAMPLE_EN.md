# MindScience Deployment Service - API Examples

This document provides examples of how to use curl commands to call the MindScience deployment service APIs.

## 1. Load Model

Use curl to load a model:

```bash
curl -X POST "http://localhost:8001/mindscience/deploy/load_model" \
  -H "Content-Type: multipart/form-data" \
  -F "model_name=your_model" \
  -F "model_file=@/path/to/your/model_file.zip"
```

The directory structure of `model_file.zip` should be:

```bash
model_file.zip
├── your_model_1.mindir
├── your_model_2.mindir
├── ...
└── your_model_n.mindir
```

If you only want to load a model from local files (without uploading), just provide `model_name`:

```bash
curl -X POST "http://localhost:8001/mindscience/deploy/load_model" \
  -H "Content-Type: multipart/form-data" \
  -F "model_name=your_local_model"
```

## 2. Unload Model

Unload the currently loaded model:

```bash
curl -X POST "http://localhost:8001/mindscience/deploy/unload_model"
```

## 3. Inference

Perform inference on a dataset. `task_type` specifies which model in `model_file.zip` to use for inference, and its value should be less than the number of models in `model_file.zip`:

```bash
curl -X POST "http://localhost:8001/mindscience/deploy/infer" \
  -H "Content-Type: multipart/form-data" \
  -F "dataset=@/path/to/your/dataset.h5" \
  -F "task_type=0"
```

This will return a task_id that can be used to check the status of the inference request.

## 4. Task Status Query

Check the status of an inference task (replace {task_id} with the actual task ID returned from the inference API):

```bash
curl -X GET "http://localhost:8001/mindscience/deploy/query_status/{task_id}"
```

For example:

```bash
curl -X GET "http://localhost:8001/mindscience/deploy/query_status/123e4567-e89b-12d3-a456-426614174000"
```

## 5. Result Download

Download the results of a completed inference task (replace {task_id} with the actual task ID):

```bash
curl -X GET "http://localhost:8001/mindscience/deploy/query_results/{task_id}" -o "results.h5"
```

The `-o` flag saves the response to a file with the specified filename. For example:

```bash
curl -X GET "http://localhost:8001/mindscience/deploy/query_results/123e4567-e89b-12d3-a456-426614174000" -o "results.h5"
```

This will download the result file named `results.h5`.

## 6. Health Check

Check the health status of the deployment service:

```bash
curl -X GET "http://localhost:8001/mindscience/deploy/health_check"
```

## Important Notes

- According to ServerConfig, the default port might be 8001, but you should check the `src/config.py` file to confirm the exact port.
- Ensure the deployment service is running before making requests.
- Model inference tasks are processed asynchronously in the background, so check the task status before attempting to download results.
- There are limits on the number of concurrent requests (configured in DeployConfig).
