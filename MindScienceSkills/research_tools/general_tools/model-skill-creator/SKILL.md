---
name: model-skill-creator
description: Create new model skills from templates. Use when you need to generate a new skill for any ML model, AI model, or pretrained model. You should read template files from the template directory and generate a complete skill directory with SKILL.md following repository conventions.
metadata:
    skill-author: MindSpore Science Team
---

# Model Skill Creator

## Overview

This skill helps you create new model skills from templates. When you need to use a model's functionality, you should use this skill to create a complete skill directory with a properly formatted SKILL.md file following the repository conventions. The generated skill can then be used to invoke that model for various tasks.

## When to Use

You should use this skill when:
- You need to create a new skill for a pretrained model (e.g., transformers, diffusers, sentence-transformers, llama, Stable Diffusion)
- You need to document a custom model or AI model for usage
- You want to standardize model documentation across the repository
- You need to generate skill documentation for any callable model

## Skill Directory Structure

A complete model skill should have the following directory structure:

```
model-skill-name/
├── SKILL.md (required)
│   ├── YAML frontmatter (name, description required)
│   └── Markdown instructions
├── references/ (optional)
│   ├── model_details.md
│   ├── api_reference.md
│   └── examples.md
├── scripts/ (optional)
│   ├── __init__.py
│   └── helper_script.py
└── assets/ (optional)
    └── example_output.png
```

### Directory Structure Explanation

| Path | Required | Description |
|------|----------|-------------|
| `SKILL.md` | Required | Main skill file with model documentation |
| `references/` | Optional | Detailed docs loaded into context as needed |
| `scripts/` | Optional | Executable code for deterministic tasks |
| `assets/` | Optional | Files used in outputs (templates, images) |

## How It Works

### Step 1: Prepare Model Information

**IMPORTANT:** All sections in the generated SKILL.md should focus only on **inference-related** functionality. Do not include training, fine-tuning, evaluation, or other non-inference scenarios. The skill should document how to use the model for inference/prediction tasks only.

You should provide the following information. **Tip:** You can often obtain this information by reading the model's GitHub README.md, official documentation, or HuggingFace model card.

**Recommended research approach:**
1. Check the model's GitHub repository README.md for installation, usage, and examples
2. Check the official documentation website for API details
3. Check HuggingFace model card (if available) for model metadata
4. Extract relevant information from these sources to fill the template

**Information to provide:**

1. **Model Basic Info** (required)
   - Model name (e.g., `llama3-8b`, `stable-diffusion-xl`, `bert-base`)
   - Short description (1-2 sentences)
   - Model type/category (e.g., LLM, text-to-image, embedding)

2. **When to Use** (required)
   - Primary application scenarios
   - Secondary application scenarios (optional)

3. **Dataset Acquisition and Processing** (required)
   - Data format requirements
   - Data size recommendations
   - Data source or acquisition methods
   - Preprocessing steps (optional)

4. **Environment Configuration** (required)
   - Required Python packages
   - Python version requirement
   - Hardware requirements (GPU/NPU, memory)
   - Disk space requirements (optional)

5. **Usage Limitations and Notes** (required)
   - Functional limitations
   - Performance limitations
   - Scale limitations
   - Important notes (optional)
   - License agreement (optional)

6. **Model Invocation Guide** (required)
   - Model initialization code
   - Inference code
   - Result post-processing (optional)

7. **Reference Resources** (optional)
   - Official documentation links
   - Related tutorials
   - Community support links

### Step 2: Generate Skill Directory

**🚨 CRITICAL REQUIREMENT - MANDATORY:** If any code example in the generated SKILL.md calls functions from other files or executes external scripts, you **MUST** place those code files and execution scripts into the `scripts/` directory. This is a **mandatory requirement** for skill generation.

You should generate the complete skill directory with:

- A new directory under `/path/to/<model-name>/`
- A properly formatted `SKILL.md` file by referencing `template/SKILL.md`:
  - Proper frontmatter (name, description)
  - All required sections populated with provided information
  - Formatted code examples
  - Reference resources section (optional)
- Optional `references/` directory (for complex models)
- Optional `scripts/` directory (if helper scripts are needed)

**Regarding scripts directory:**
- If you create a `scripts/` directory with helper scripts, you should document their usage scenarios in the generated SKILL.md (e.g., in a dedicated section or within relevant code examples)
- If the helper script provides functionality that overlaps with code examples shown in SKILL.md, you should only include the script invocation method in SKILL.md (e.g., `from scripts import helper_script; helper_script.run()`) and remove the duplicate inline code

**Regarding references and assets directories:**
- If you create a `references/` directory with additional documentation files, you should document in the generated SKILL.md when to read those files (e.g., "For advanced API details, see references/api_reference.md") and how to access them
- If you create an `assets/` directory with supporting files (templates, images, etc.), you should document in the generated SKILL.md where and how to use those assets (e.g., "Output templates are available in assets/template.json")

### Step 3: Output Location

The generated skill will be saved to:
```
/path/to/<model-name>/
└── SKILL.md
```

## Required Information Format

You should organize the generated SKILL.md with the following sections, referencing `template/SKILL.md` for the exact format:

### 1. Overview
Provide a comprehensive overview of the model's functionality, applicable scenarios, and core capabilities.

### 2. When to Use
List the primary application scenarios. Mark secondary scenarios as "(optional)".

### 3. Dataset Acquisition and Processing
Describe data format, size, source, and preprocessing steps. Mark optional steps as "(optional)".

### 4. Environment Configuration and Dependencies
Specify Python version, hardware requirements (GPU/NPU), memory, and disk space. Include dependency installation commands.

### 5. Usage Limitations and Notes
List functional, performance, and scale limitations. Include important notes and license information (mark as optional if not applicable).

### 6. Model Invocation Guide
Provide code examples for model initialization, inference, and result post-processing.

### 7. Reference Resources (optional)
Include links to official documentation, tutorials, and community support.

## Best Practices

**When creating a skill:**
- You should provide as much detail as possible for each section
- You should include actual code snippets if available (read from model README if possible)
- You should verify license information before generation
- You should include official documentation links for reference
- **Recommended:** You should read the model's GitHub README.md or official docs first to get accurate information

**For generated skills:**
- Keep descriptions concise but informative
- Include working code examples
- List all required dependencies
- Specify exact version requirements if needed
- **IMPORTANT:** All code examples in SKILL.md must be tested and verified to work before writing. Run the code using Python REPL or terminal to verify it executes correctly. If environment issues prevent validation (e.g., missing GPU, CUDA, or platform-specific dependencies), add clear comments in the code explaining the environment requirements and keep the code in SKILL.md anyway.

## Output

You should create:
1. A new directory under `/path/to/<model-name>/`
2. A properly formatted `SKILL.md` file
3. Optional `references/` directory (for complex models)
4. Optional `scripts/` directory (if helper code is needed)

After generation, the new skill can be used to invoke the model for various tasks.

## References

- Template file: `template/SKILL.md`
- Example skills: See any skill in a given directory (e.g., `MindScienceSkills/esm`, `MindScienceSkills/transformers`, `MindScienceSkills/diffdock`)
- Reference: `MindScienceSkills/research_tools/general_tools/skill-creator/SKILL.md` for general skill creation patterns