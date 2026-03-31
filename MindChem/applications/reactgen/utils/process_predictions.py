import json

# --- 1. 配置：请修改这里 ---

# 将你 8 卡的JSON 文件路径/文件名放在这里
folder_name = "best_bs12_lr1e-6_20251107_110237"
JSON_FILES = [
    f"./logs/forward/full_inference_results/qwen2.5-1.5b/{folder_name}/predictions_rank_0.json",
    f"./logs/forward/full_inference_results/qwen2.5-1.5b/{folder_name}/predictions_rank_1.json",
    f"./logs/forward/full_inference_results/qwen2.5-1.5b/{folder_name}/predictions_rank_2.json",
    f"./logs/forward/full_inference_results/qwen2.5-1.5b/{folder_name}/predictions_rank_3.json",
    f"./logs/forward/full_inference_results/qwen2.5-1.5b/{folder_name}/predictions_rank_4.json",
    f"./logs/forward/full_inference_results/qwen2.5-1.5b/{folder_name}/predictions_rank_5.json",
    f"./logs/forward/full_inference_results/qwen2.5-1.5b/{folder_name}/predictions_rank_6.json",
    f"./logs/forward/full_inference_results/qwen2.5-1.5b/{folder_name}/predictions_rank_7.json",
]

# 你希望保存的最终汇总文件名
OUTPUT_FILE = f"./logs/forward/full_inference_results/qwen2.5-1.5b/{folder_name}/eval_metrics_final.json"
MERGED_PREDICTIONS_FILE = f"./logs/forward/full_inference_results/qwen2.5-1.5b/{folder_name}/forward_predictions.json"
# -----------------------------

def analyze_json_files(file_list):
    """
    遍历所有 JSON 文件，汇总并计算所有指标。
    """
    
    # 初始化总计数器
    total_predictions = 0
    valid_predictions = 0
    exact_matches = 0

    print(f"开始处理 {len(file_list)} 个文件...")

    # --- 2. 遍历每个文件并累加数据 ---
    all_records = []
    for file_path in file_list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 使用 json.load() 因为每个文件本身是一个大的 JSON 列表
                data = json.load(f)

            all_records.extend(data)
            
            # 确保文件内容是列表
            if not isinstance(data, list):
                print(f"⚠️ 警告: {file_path} 不是一个列表, 已跳过。")
                continue
            
            print(f"  正在处理 {file_path} (包含 {len(data)} 条记录)...")

            # 遍历文件中的每一条记录
            for record in data:
                # 每一条记录都算作一个 "prediction"
                total_predictions += 1
                
                # 检查 is_valid
                # 使用 .get("key") is True 是最安全的写法
                # 它可以防止因 "key" 不存在而报错，也确保只统计布尔值 True
                if record.get("is_valid") is True:
                    valid_predictions += 1
                
                # 检查 exact_match
                if record.get("exact_match") is True:
                    exact_matches += 1

        except FileNotFoundError:
            print(f"❌ 错误: 找不到文件 {file_path}, 已跳过。")
        except json.JSONDecodeError:
            print(f"❌ 错误: {file_path} 不是一个有效的 JSON 文件, 已跳过。")
        except Exception as e:
            print(f"❌ 处理 {file_path} 时发生未知错误: {e}")

    print("\n...所有文件处理完毕, 开始计算最终指标。")

    # --- 3. 计算最终的派生指标 ---
    
    # 按照你的定义计算
    invalid_predictions = total_predictions - valid_predictions
    
    # (安全检查) 处理除零错误，防止 total_predictions 为 0
    if total_predictions > 0:
        validity_rate = valid_predictions / total_predictions
        top1_accuracy = exact_matches / total_predictions
    else:
        validity_rate = 0.0
        top1_accuracy = 0.0
        print("⚠️ 警告: 'total_predictions' 为 0。")

    # (安全检查) 处理除零错误，防止 valid_predictions 为 0
    if valid_predictions > 0:
        valid_accuracy = exact_matches / valid_predictions
    else:
        valid_accuracy = 0.0
        if exact_matches > 0:
            print("⚠️ 警告: 有 'exact_matches' 但 'valid_predictions' 为 0。")

    # --- 4. 整理最终的输出结果 ---
    final_results = {
        "total_predictions": total_predictions,
        "valid_predictions": valid_predictions,
        "exact_matches": exact_matches,
        "validity_rate": validity_rate,
        "top1_accuracy": top1_accuracy,
        "valid_accuracy": valid_accuracy,
        "invalid_predictions": invalid_predictions
    }

    # --- 5. 写入到文件 ---
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # indent=2 使输出的 JSON 文件格式化，易于阅读
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*30)
        print(f"✅ 成功！汇总结果已保存到: {OUTPUT_FILE}")
        print("="*30)

        with open(MERGED_PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_records, f, indent=2, ensure_ascii=False)

        # 也在控制台打印一份结果
        print(json.dumps(final_results, indent=2))
        
    except IOError as e:
        print(f"❌ 致命错误: 无法写入输出文件 {OUTPUT_FILE}。错误: {e}")


if __name__ == "__main__":
    if JSON_FILES[0] == "file1.json":
        print("✋ 请先修改脚本顶部的 `JSON_FILES` 列表，填入你真实的文件路径！")
    else:
        analyze_json_files(JSON_FILES)