#!/usr/bin/env python3
"""
快速验证本地Llama 3.2 3B模型是否可以正确加载
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def verify_model(model_path: str = "/home/mpcblock/models/Llama-3.2-3B"):
    """验证本地模型是否可以正确加载"""
    
    print("=" * 60)
    print("验证本地Llama 3.2 3B模型")
    print("=" * 60)
    print(f"\n模型路径: {model_path}")
    
    # 检查路径是否存在
    if not os.path.exists(model_path):
        print(f"\n❌ 错误: 模型路径不存在: {model_path}")
        print("\n请检查:")
        print("1. 路径是否正确")
        print("2. 模型是否已下载到该路径")
        return False
    
    if not os.path.isdir(model_path):
        print(f"\n❌ 错误: 路径不是目录: {model_path}")
        return False
    
    # 检查必要文件
    required_files = ["config.json"]
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  警告: 缺少以下文件: {', '.join(missing_files)}")
        print("代码会尝试自动下载缺失的配置文件")
    else:
        print("\n✅ 模型目录结构检查通过")
    
    # 检查GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}")
    
    if device == "cuda":
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU显存: {gpu_memory:.2f} GB")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
    # 尝试加载tokenizer
    print("\n[步骤1] 加载tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=False,  # 允许下载缺失的配置文件
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer加载成功")
    except Exception as e:
        print(f"❌ Tokenizer加载失败: {e}")
        return False
    
    # 尝试加载模型（FP16，无量化）
    print("\n[步骤2] 加载模型（FP16，无量化）...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            local_files_only=False,
            trust_remote_code=True,
        )
        
        if device == "cpu":
            model = model.to(device)
        
        print("✅ 模型加载成功（FP16）")
        
        # 检查参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   参数量: {total_params:,}")
        print(f"   模型大小: {total_params * 2 / (1024**3):.2f} GB (FP16)")
        
        # 检查显存使用
        if device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"   GPU显存使用: {memory_allocated:.2f} GB (已分配)")
            print(f"   GPU显存预留: {memory_reserved:.2f} GB (已预留)")
            
            if memory_reserved > gpu_memory * 0.9:
                print("\n⚠️  警告: 显存使用接近上限，建议使用4-bit量化")
            else:
                print("\n✅ 显存使用正常，可以继续使用FP16")
        
        # 简单测试
        print("\n[步骤3] 测试模型推理...")
        test_text = "Hello, how are you?"
        inputs = tokenizer(test_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=10,
                do_sample=False,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   输入: {test_text}")
        print(f"   输出: {generated}")
        print("✅ 模型推理测试成功")
        
        print("\n" + "=" * 60)
        print("✅ 模型验证完成！可以用于实验")
        print("=" * 60)
        
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n❌ OOM错误: {e}")
            print("\n建议:")
            print("1. 使用4-bit量化: --quantization-bits 4")
            print("2. 关闭其他占用GPU的程序")
            print("3. 检查模型是否正确下载")
        else:
            print(f"\n❌ 模型加载失败: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        return False


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/mpcblock/models/Llama-3.2-3B"
    success = verify_model(model_path)
    sys.exit(0 if success else 1)

