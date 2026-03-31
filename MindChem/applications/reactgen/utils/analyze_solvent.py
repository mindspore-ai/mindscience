import ast
import matplotlib.pyplot as plt

def check_data_order(file_path):
    print(f"正在分析数据顺序: {file_path}")
    
    y = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i > 50000: break # 只看前5万行够了
            parts = line.split('\t')
            if len(parts) < 3: continue
            
            try:
                # 假设 parts[2] 是 [3, 10]
                labels = ast.literal_eval(parts[2])
                if labels:
                    y.append(labels[0]) # 取第一个溶剂ID作为代表
                else:
                    y.append(-1)
            except:
                continue
                
    # 如果数据是打乱的，y 应该是杂乱无章的散点图
    # 如果数据是有序的，y 会呈现出明显的台阶状或条纹状
    print("前 100 个样本的 ID:", y[:100])
    
    # 简单的文本可视化
    print("\n简单可视化前 50 行 ID:")
    for id_val in y[:50]:
        print(f"{id_val} " * int(id_val > 0), end="") 
        print("|")

if __name__ == "__main__":
    check_data_order("./data/solvent/train.txt")