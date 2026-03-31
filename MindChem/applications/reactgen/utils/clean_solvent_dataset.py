import ast
import os

def process_dataset(input_file, output_file):
    print(f"正在处理: {input_file} -> {output_file}")
    
    total_lines = 0
    modified_lines = 0
    removed_51_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        # 1. 处理表头 (如果有的话)
        # 读取第一行，检查是否是表头
        first_line = f_in.readline()
        if not first_line:
            return
            
        if "reaction" in first_line.lower() or "solvents" in first_line.lower():
            f_out.write(first_line)
        else:
            # 如果第一行不是表头，回退指针重新处理（或者视情况而定）
            # 这里假设第一行就是表头
            pass 

        # 2. 逐行处理数据
        # 如果刚才读取的是表头，循环会从第二行开始；如果不是，这里需要调整
        # 为简单起见，假设第一行必须是表头，或者手动将非表头数据补回去
        # (通常 datasets 都有 header)
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            total_lines += 1
            parts = line.split('\t')
            
            if len(parts) < 3:
                f_out.write(line + '\n')
                continue
                
            reaction = parts[0]
            solvents_str = parts[1]
            labels_str = parts[2]
            
            try:
                # 安全解析
                solvents_list = ast.literal_eval(solvents_str)
                labels_list = ast.literal_eval(labels_str)
                
                # --- 特殊情况处理：原本就是无溶剂 ---
                # 如果原本就是 [], [0]，直接保留，不做处理
                if len(solvents_list) == 0 and labels_list == [0]:
                    f_out.write(line + '\n')
                    continue

                # --- 核心清洗逻辑 ---
                clean_solvents = []
                clean_labels = []
                seen_ids = set() # 用于去重 (例如把 [8, 8] 变成 [8])
                
                has_change = False
                
                for s, l in zip(solvents_list, labels_list):
                    l_int = int(l)
                    
                    # 1. 剔除 51 号溶剂
                    if l_int == 51:
                        has_change = True
                        removed_51_count += 1
                        continue
                    
                    # 2. 去重逻辑 (如果你需要保留重复，注释掉这就行)
                    if l_int in seen_ids:
                        has_change = True
                        continue
                    
                    # 保留有效数据
                    seen_ids.add(l_int)
                    clean_solvents.append(s)
                    clean_labels.append(l_int)
                
                # --- 结果组装 ---
                if len(clean_labels) == 0:
                    # 如果清洗完变空了 (比如原数据只有51，或者全是重复)，
                    # 必须转为标准的"无溶剂"格式: solvents=[], labels=[0]
                    new_line = f"{reaction}\t[]\t[0]\n"
                    modified_lines += 1
                elif has_change:
                    # 如果有变化 (剔除了51或去重了)，写入新数据
                    new_line = f"{reaction}\t{str(clean_solvents)}\t{str(clean_labels)}\n"
                    modified_lines += 1
                else:
                    # 如果没变化，写回原行 (防止 str() 改变格式微小差异)
                    f_out.write(line + '\n')
                    continue

                f_out.write(new_line)
                
            except Exception as e:
                print(f"行 {total_lines} 解析失败: {e}")
                f_out.write(line + '\n') # 出错保留原样

    print("-" * 30)
    print(f"处理完成！")
    print(f"总行数: {total_lines}")
    print(f"修改行数: {modified_lines}")
    print(f"剔除 51 号标签次数: {removed_51_count}")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    # 配置路径
    data_dir = "./data/solvent" # 你的数据目录
    
    # 需要处理的文件列表
    files = ["train.txt", "valid.txt", "test.txt"]
    
    for filename in files:
        input_path = os.path.join(data_dir, filename)
        output_path = os.path.join(data_dir, f"clean_{filename}") # 输出为 clean_train.txt 等
        
        if os.path.exists(input_path):
            process_dataset(input_path, output_path)
        else:
            print(f"跳过: {input_path} 不存在")