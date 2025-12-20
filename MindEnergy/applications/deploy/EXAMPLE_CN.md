# MindScience 部署服务 - API 示例

本文档提供如何使用 curl 命令调用 MindScience 部署服务 API 的示例。

## 1. 加载模型

使用 curl 加载模型：

```bash
curl -X POST "http://localhost:8001/mindscience/deploy/load_model" \
  -H "Content-Type: multipart/form-data" \
  -F "model_name=your_model" \
  -F "model_file=@/path/to/your/model_file.zip"
```

`model_file.zip` 的目录格式为：

```bash
model_file.zip
├── your_model_1.mindir
├── your_model_2.mindir
├── ...
└── your_model_n.mindir
```

如果只想从本地文件加载模型（不上传），只需提供 `model_name`：

```bash
curl -X POST "http://localhost:8001/mindscience/deploy/load_model" \
  -H "Content-Type: multipart/form-data" \
  -F "model_name=your_local_model"
```

## 2. 卸载模型

卸载当前已加载的模型：

```bash
curl -X POST "http://localhost:8001/mindscience/deploy/unload_model"
```

## 3. 推理

对数据集执行推理，`task_type` 指定选择 `model_file.zip` 中的哪个模型进行推理，其取值需小于 `model_file.zip` 中的模型数量：

```bash
curl -X POST "http://localhost:8001/mindscience/deploy/infer" \
  -H "Content-Type: multipart/form-data" \
  -F "dataset=@/path/to/your/dataset.h5" \
  -F "task_type=0"
```

这将返回一个 task_id，可用于检查推理请求的状态。

## 4. 任务状态查询

检查推理任务的状态（将 {task_id} 替换为从推理API返回的实际任务 ID）：

```bash
curl -X GET "http://localhost:8001/mindscience/deploy/query_status/{task_id}"
```

例如：

```bash
curl -X GET "http://localhost:8001/mindscience/deploy/query_status/123e4567-e89b-12d3-a456-426614174000"
```

## 5. 结果下载

下载已完成的推理任务的结果（将 {task_id} 替换为实际的任务 ID）：

```bash
curl -X GET "http://localhost:8001/mindscience/deploy/query_results/{task_id}" -o "results.h5"
```

`-o` 标志将响应保存为指定文件名的文件。例如：

```bash
curl -X GET "http://localhost:8001/mindscience/deploy/query_results/123e4567-e89b-12d3-a456-426614174000" -o "results.h5"
```

这将下载名为 `results.h5` 的结果文件。

## 6. 健康检查

检查部署服务的健康状态：

```bash
curl -X GET "http://localhost:8001/mindscience/deploy/health_check"
```

## 重要说明

- 根据 ServerConfig，默认端口可能是 8001，但应检查 `src/config.py` 文件以确认确切端口。
- 发出请求之前，请确保部署服务正在运行。
- 模型推理任务在后台异步处理，因此在尝试下载结果之前需要检查任务状态。
- 对于并发请求数有限制（在 DeployConfig 中配置）。
