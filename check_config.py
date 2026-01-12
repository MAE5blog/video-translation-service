#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理模块
从config.ini读取配置，支持环境变量覆盖
"""

import os
import configparser
from pathlib import Path


class Config:
    """配置管理器"""

    def __init__(self, config_file='config.ini'):
        self.config_file = Path(config_file)
        self.config = configparser.ConfigParser()

        # 尝试加载配置文件
        if self.config_file.exists():
            self.config.read(self.config_file, encoding='utf-8')
        else:
            print(f"警告: 配置文件不存在 {config_file}")
            print(f"请复制 config.template.ini 为 config.ini 并填写配置")

    def get(self, section, key, default=None):
        """获取配置值，支持环境变量覆盖"""
        # 环境变量名格式: SECTION_KEY (大写)
        env_key = f"{section.upper()}_{key.upper()}"

        # 优先使用环境变量
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value

        # 然后使用配置文件
        try:
            return self.config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default

    def get_bool(self, section, key, default=False):
        """获取布尔配置"""
        value = self.get(section, key, str(default))
        return value.lower() in ('true', '1', 'yes', 'on')

    def get_int(self, section, key, default=0):
        """获取整数配置"""
        value = self.get(section, key, str(default))
        try:
            return int(value)
        except ValueError:
            return default

    def get_float(self, section, key, default=0.0):
        """获取浮点数配置"""
        value = self.get(section, key, str(default))
        try:
            return float(value)
        except ValueError:
            return default

    # API配置
    @property
    def deepseek_api_key(self):
        """DeepSeek API密钥"""
        return self.get('API', 'deepseek_api_key', os.getenv('DEEPSEEK_API_KEY', ''))

    # 服务配置
    @property
    def service_host(self):
        return self.get('Service', 'host', '127.0.0.1')

    @property
    def service_port(self):
        return self.get_int('Service', 'port', 50515)

    # 模型配置
    @property
    def asr_model_size(self):
        return self.get('Models', 'asr_model_size', 'reazonspeech')

    @property
    def translation_model(self):
        return self.get('Models', 'translation_model', 'SakuraLLM/Sakura-4B-Qwen3-Base-v2')

    @property
    def use_gpu(self):
        return self.get_bool('Models', 'use_gpu', True)

    @property
    def beam_size(self):
        return self.get_int('Models', 'beam_size', 3)

    # GPU运行配置
    @property
    def clear_cuda_cache_before_tasks(self):
        """是否在 GPU 重任务前清理 CUDA 显存缓存（torch.cuda.empty_cache）"""
        return self.get_bool('GPU', 'clear_cuda_cache_before_tasks', False)

    # ASR 运行配置
    @property
    def asr_transformers_chunk_sec(self):
        """Transformers ASR 内部分块秒数（仅对 transformers 后端生效；0=禁用）"""
        return self.get_int('ASR', 'transformers_chunk_sec', 30)

    @property
    def asr_transformers_stride_sec(self):
        """Transformers ASR 分块重叠秒数（仅对 transformers 后端生效）"""
        return self.get_float('ASR', 'transformers_stride_sec', 5.0)

    # GGUF/llama.cpp 配置
    @property
    def gguf_n_ctx(self):
        return self.get_int('GGUF', 'n_ctx', 4096)

    @property
    def gguf_n_threads(self):
        return self.get_int('GGUF', 'n_threads', 4)

    @property
    def gguf_n_batch(self):
        return self.get_int('GGUF', 'n_batch', 256)

    @property
    def gguf_n_gpu_layers(self):
        return self.get_int('GGUF', 'n_gpu_layers', -1)

    @property
    def gguf_temperature(self):
        return self.get_float('GGUF', 'temperature', 0.1)

    @property
    def gguf_top_p(self):
        return self.get_float('GGUF', 'top_p', 0.9)

    @property
    def gguf_repeat_penalty(self):
        return self.get_float('GGUF', 'repeat_penalty', 1.05)

    # 翻译配置
    @property
    def default_target_language(self):
        return self.get('Translation', 'default_target_language', 'zh')

    @property
    def use_deepseek_polish(self):
        return self.get_bool('Translation', 'use_deepseek_polish', False)

    # 音频预处理
    @property
    def enable_vocal_separation(self):
        """是否启用人声分离（Demucs）"""
        return self.get_bool('Audio', 'enable_vocal_separation', False)

    @property
    def vocal_separation_model(self):
        """Demucs 模型名"""
        return self.get('Audio', 'vocal_separation_model', 'htdemucs')

    @property
    def vocal_separation_device(self):
        """人声分离设备：auto / cpu / cuda"""
        return self.get('Audio', 'vocal_separation_device', 'auto')


# 全局配置实例
config = Config()

if __name__ == '__main__':
    # 测试配置
    print("配置测试：")
    print(f"DeepSeek API密钥: {'已设置' if config.deepseek_api_key else '未设置'}")
    print(f"服务地址: {config.service_host}:{config.service_port}")
    print(f"ASR模型: {config.asr_model_size}")
    print(f"翻译模型: {config.translation_model}")
    print(f"使用GPU: {config.use_gpu}")
    print(f"beam_size: {config.beam_size}")
    print(f"GPU任务前清理显存缓存: {config.clear_cuda_cache_before_tasks}")
    print(f"ASR transformers 分块秒数: {config.asr_transformers_chunk_sec}")
    print(f"ASR transformers 分块重叠秒数: {config.asr_transformers_stride_sec}")
    print(f"GGUF n_ctx: {config.gguf_n_ctx}")
    print(f"GGUF n_threads: {config.gguf_n_threads}")
    print(f"GGUF n_batch: {config.gguf_n_batch}")
    print(f"GGUF n_gpu_layers: {config.gguf_n_gpu_layers}")
    print(f"GGUF temperature: {config.gguf_temperature}")
    print(f"GGUF top_p: {config.gguf_top_p}")
    print(f"GGUF repeat_penalty: {config.gguf_repeat_penalty}")
    print(f"默认目标语言: {config.default_target_language}")
    print(f"使用DeepSeek润色: {config.use_deepseek_polish}")
    print(f"启用人声分离: {config.enable_vocal_separation}")
    print(f"人声分离模型: {config.vocal_separation_model}")
    print(f"人声分离设备: {config.vocal_separation_device}")
