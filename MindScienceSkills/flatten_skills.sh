#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# flatten_skills.sh - 将MindScienceSkills中的skills平铺到新目录
#
# 用法:
#   ./flatten_skills.sh                                         # 平铺所有skills到默认目录
#   ./flatten_skills.sh -o /path/to/output                      # 平铺所有skills到指定目录
#   ./flatten_skills.sh biology                                 # 只平铺biology目录下的skills
#   ./flatten_skills.sh earth_sciences/meteorology              # 平铺指定路径下的skills
#   ./flatten_skills.sh "biology, earth_sciences/meteorology"   # 平铺多个目录
#   ./flatten_skills.sh -o /path/to/output biology              # 平铺指定目录到指定输出

set -e

# 配置
SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
# 默认目标目录：和MindScienceSkills平级的MindScienceSkills_flattened
TARGET_DIR="$(dirname "$SOURCE_DIR")/MindScienceSkills_flattened"

# 解析参数
FLATTEN_DIRS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            TARGET_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项] [目录1,目录2,...]"
            echo ""
            echo "选项:"
            echo "  -o, --output DIR    指定输出目录 (默认: ../MindScienceSkills_flattened)"
            echo "  -h, --help          显示帮助"
            echo ""
            echo "示例:"
            echo "  $0                              # 平铺所有skills到默认目录"
            echo "  $0 -o /path/to/output           # 平铺所有skills到指定目录"
            echo "  $0 biology                      # 只平铺biology目录下的skills"
            echo "  $0 earth_sciences/meteorology   # 平铺earth_sciences/meteorology下的skills"
            echo "  $0 'biology, earth_sciences'    # 平铺多个目录"
            echo "  $0 -o /path/to/output biology   # 平铺指定目录到指定输出"
            exit 0
            ;;
        *)
            FLATTEN_DIRS="$1"
            shift
            ;;
    esac
done

# 创建目标目录
mkdir -p "$TARGET_DIR"

echo "源目录: $SOURCE_DIR"
echo "目标目录: $TARGET_DIR"
echo "指定平铺目录: ${FLATTEN_DIRS:-全部}"
echo ""

# 将逗号分隔的目录转换为数组
IFS=',' read -ra FLATTEN_ARRAY <<< "$FLATTEN_DIRS"

# 确定搜索路径：如果指定了目录，则只搜索这些目录；否则搜索全部
SEARCH_PATHS=()
if [ -n "$FLATTEN_DIRS" ]; then
    for dir in "${FLATTEN_ARRAY[@]}"; do
        dir=$(echo "$dir" | xargs)
        if [[ -n "$dir" ]]; then
            SEARCH_PATHS+=("$SOURCE_DIR/$dir")
        fi
    done
else
    SEARCH_PATHS=("$SOURCE_DIR")
fi

# 查找所有包含SKILL.md的目录
for search_path in "${SEARCH_PATHS[@]}"; do
    if [ ! -d "$search_path" ]; then
        echo "警告: 目录不存在: $search_path"
        continue
    fi

    find "$search_path" -name "SKILL.md" -type f | while read -r skill_file; do
        skill_dir="$(dirname "$skill_file")"

        # 获取skill目录相对于SOURCE_DIR的路径
        relative_path="${skill_dir#$SOURCE_DIR/}"

        # 新名称：只使用原来的目录名（最后一级的目录名）
        new_name="$(basename "$skill_dir")"

        # 检查SKILL.md里的name是否和目录名一致
        skill_name=$(sed -n 's/^name: *//p' "$skill_file" | head -1)
        if [[ "$skill_name" != "$new_name" ]]; then
            continue
        fi

        target_path="$TARGET_DIR/$new_name"

        # 复制目录
        echo "复制: $relative_path -> $new_name"
        cp -r "$skill_dir" "$target_path"
    done
done

echo ""
echo "完成！所有skills已平铺到 $TARGET_DIR"
echo "总计: $(ls -1 "$TARGET_DIR" | wc -l) 个skills"
