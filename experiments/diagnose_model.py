#!/usr/bin/env python3
"""
诊断本地模型目录，检查文件结构和配置
"""

import os
import sys
import json

def diagnose_model(model_path: str = "/home/mpcblock/models/Llama-3.2-3B"):
    """诊断模型目录"""
    
    print("=" * 60)
    print("模型目录诊断")
    print("=" * 60)
    print(f"\n模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"\n❌ 路径不存在: {model_path}")
        return False
    
    if not os.path.isdir(model_path):
        print(f"\n❌ 不是目录: {model_path}")
        return False
    
    print("\n[检查1] 目录内容:")
    files = os.listdir(model_path)
    for f in sorted(files):
        file_path = os.path.join(model_path, f)
        size = os.path.getsize(file_path) / (1024**2) if os.path.isfile(file_path) else 0
        file_type = "DIR" if os.path.isdir(file_path) else "FILE"
        print(f"  {file_type:4s} {f:40s} {size:8.2f} MB" if size > 0 else f"  {file_type:4s} {f}")
    
    # 检查关键文件
    print("\n[检查2] 关键文件:")
    key_files = {
        "config.json": "模型配置",
        "tokenizer.json": "Tokenizer数据",
        "tokenizer_config.json": "Tokenizer配置",
        "model.safetensors": "模型权重（safetensors格式）",
        "pytorch_model.bin": "模型权重（bin格式）",
        "model-*.safetensors": "分片模型权重",
    }
    
    found_files = {}
    for key_file, desc in key_files.items():
        if "*" in key_file:
            # Check for pattern
            pattern = key_file.replace("*", "")
            matching = [f for f in files if pattern in f]
            if matching:
                found_files[key_file] = (True, desc, matching)
                print(f"  ✅ {key_file}: {desc} (找到: {', '.join(matching[:3])})")
        else:
            file_path = os.path.join(model_path, key_file)
            exists = os.path.exists(file_path)
            found_files[key_file] = (exists, desc, None)
            if exists:
                print(f"  ✅ {key_file}: {desc}")
            else:
                print(f"  ❌ {key_file}: {desc} (缺失)")
    
    # 检查config.json内容
    print("\n[检查3] config.json内容:")
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"  ✅ config.json格式正确")
            print(f"  - model_type: {config.get('model_type', 'NOT FOUND')}")
            print(f"  - vocab_size: {config.get('vocab_size', 'NOT FOUND')}")
            print(f"  - hidden_size: {config.get('hidden_size', 'NOT FOUND')}")
            print(f"  - num_attention_heads: {config.get('num_attention_heads', 'NOT FOUND')}")
            print(f"  - num_hidden_layers: {config.get('num_hidden_layers', 'NOT FOUND')}")
        except Exception as e:
            print(f"  ❌ config.json读取失败: {e}")
    else:
        print("  ❌ config.json不存在")
    
    # 检查tokenizer_config.json
    print("\n[检查4] tokenizer_config.json内容:")
    tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        try:
            with open(tokenizer_config_path, 'r') as f:
                tokenizer_config = json.load(f)
            print(f"  ✅ tokenizer_config.json格式正确")
            print(f"  - tokenizer_class: {tokenizer_config.get('tokenizer_class', 'NOT FOUND')}")
            print(f"  - auto_map: {tokenizer_config.get('auto_map', 'NOT FOUND')}")
        except Exception as e:
            print(f"  ❌ tokenizer_config.json读取失败: {e}")
    else:
        print("  ⚠️  tokenizer_config.json不存在（可能使用默认配置）")
    
    # 检查tokenizer.json
    print("\n[检查5] tokenizer.json:")
    tokenizer_json_path = os.path.join(model_path, "tokenizer.json")
    if os.path.exists(tokenizer_json_path):
        size = os.path.getsize(tokenizer_json_path) / (1024**2)
        print(f"  ✅ tokenizer.json存在 ({size:.2f} MB)")
    else:
        print("  ❌ tokenizer.json不存在")
    
    # 总结
    print("\n" + "=" * 60)
    print("诊断总结:")
    print("=" * 60)
    
    has_config = found_files.get("config.json", (False,))[0]
    has_tokenizer_json = found_files.get("tokenizer.json", (False,))[0]
    has_model = (found_files.get("model.safetensors", (False,))[0] or 
                 found_files.get("pytorch_model.bin", (False,))[0] or
                 found_files.get("model-*.safetensors", (False,))[0])
    
    if has_config and has_tokenizer_json and has_model:
        print("✅ 模型目录看起来完整，应该可以加载")
    else:
        print("⚠️  模型目录可能不完整:")
        if not has_config:
            print("  - 缺少config.json")
        if not has_tokenizer_json:
            print("  - 缺少tokenizer.json")
        if not has_model:
            print("  - 缺少模型权重文件")
        print("\n建议:")
        print("1. 从HuggingFace重新下载完整模型")
        print("2. 使用: huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir /home/mpcblock/models/Llama-3.2-3B")
    
    return has_config and has_tokenizer_json and has_model


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/mpcblock/models/Llama-3.2-3B"
    success = diagnose_model(model_path)
    sys.exit(0 if success else 1)

