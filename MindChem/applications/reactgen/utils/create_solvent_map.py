import json

# 1. 设置文件名 (根据你的实际文件名修改这里)
file_path = './data/solvent/solvent_map.json'

try:
    # 2. 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 3. 删除值为 51 的项
    # 使用字典推导式创建一个新字典，保留所有 值不等于 51 的项
    original_count = len(data)
    data = {k: v for k, v in data.items() if v != 51}
    removed_count = original_count - len(data)

    # 4. 把修改后的内容写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        # indent=2 让文件保存得好看一点（有缩进），ensure_ascii=False 防止中文乱码
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"处理完成！共删除了 {removed_count} 个值为 51 的条目。")

except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}，请检查文件名是否正确。")
except json.JSONDecodeError:
    print("错误：文件格式不是有效的 JSON，请检查文件内容。")