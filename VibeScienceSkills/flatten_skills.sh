#!/bin/bash

# 配置
SOURCE_DIR="."
TARGET_DIR="flatten_VibeScienceSkills"
PREFIX_LEVEL=1  # 0: 无前缀, 1: 使用第二级前缀, 2: 使用第一级+第二级前缀

# 创建目标目录
mkdir -p "$TARGET_DIR"

# 遍历第一级目录
for first_dir in "$SOURCE_DIR"/*/; do
    if [ ! -d "$first_dir" ]; then
        continue
    fi
    
    first_name=$(basename "$first_dir")
    
    # 遍历第二级目录
    for second_dir in "$first_dir"*/; do
        if [ ! -d "$second_dir" ]; then
            continue
        fi
        
        second_name=$(basename "$second_dir")
        
        # 遍历第二级目录下的所有子目录
        for sub_dir in "$second_dir"*/; do
            if [ ! -d "$sub_dir" ]; then
                continue
            fi
            
            sub_name=$(basename "$sub_dir")
            
            # 根据前缀级别生成新名称
            if [ "$PREFIX_LEVEL" -eq 0 ]; then
                new_name="$sub_name"
            elif [ "$PREFIX_LEVEL" -eq 1 ]; then
                new_name="${second_name}_${sub_name}"
            else
                new_name="${first_name}_${second_name}_${sub_name}"
            fi
            
            # 处理重名
            target_path="$TARGET_DIR/$new_name"
            counter=1
            while [ -d "$target_path" ]; do
                if [ "$PREFIX_LEVEL" -eq 0 ]; then
                    target_path="$TARGET_DIR/${sub_name}_$counter"
                elif [ "$PREFIX_LEVEL" -eq 1 ]; then
                    target_path="$TARGET_DIR/${second_name}_${sub_name}_$counter"
                else
                    target_path="$TARGET_DIR/${first_name}_${second_name}_${sub_name}_$counter"
                fi
                ((counter++))
            done
            
            # 移动目录
            echo "移动: $sub_dir -> $target_path"
            mv "$sub_dir" "$target_path"
        done
    done
done

echo -e "\n完成！所有子目录已移动到 $TARGET_DIR 中"