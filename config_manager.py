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

    @property
    def lazy_load_models(self):
        """服务端是否启动时预加载模型（false=预加载；true=按需加载）"""
        return self.get_bool('Service', 'lazy_load_models', False)

    @property
    def manage_models(self):
        """客户端是否按需加载/卸载服务端模型（适合显存紧张环境）"""
        return self.get_bool('Service', 'manage_models', False)

    @property
    def unload_models_after_tasks(self):
        """客户端是否在 ASR/翻译完成后卸载模型释放显存"""
        return self.get_bool('Service', 'unload_models_after_tasks', False)

    # 模型配置
    @property
    def asr_model_size(self):
        return self.get('Models', 'asr_model_size', 'reazonspeech')

    @property
    def translation_model(self):
        return self.get('Models', 'translation_model', 'facebook/nllb-200-1.3B')

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
    def asr_language(self):
        """指定 ASR 音频语言（auto=自动检测）"""
        return self.get('ASR', 'language', 'auto')

    @property
    def asr_chunk_sec(self):
        """ASR 分块秒数（0=禁用）"""
        return self.get_int('ASR', 'chunk_sec', 0)

    @property
    def asr_chunk_overlap_sec(self):
        """ASR 分块重叠秒数（避免切在单词中间）"""
        return self.get_float('ASR', 'chunk_overlap_sec', 0.0)

    @property
    def asr_transformers_chunk_sec(self):
        """Transformers ASR 内部分块秒数（仅对 transformers 后端生效；0=禁用）"""
        return self.get_int('ASR', 'transformers_chunk_sec', 30)

    @property
    def asr_transformers_stride_sec(self):
        """Transformers ASR 分块重叠秒数（仅对 transformers 后端生效）"""
        return self.get_float('ASR', 'transformers_stride_sec', 5.0)

    # 翻译配置
    @property
    def default_target_language(self):
        return self.get('Translation', 'default_target_language', 'zh')

    @property
    def use_deepseek_polish(self):
        return self.get_bool('Translation', 'use_deepseek_polish', False)

    # 字幕输出
    @property
    def subtitle_format(self):
        """字幕格式：srt / ass"""
        v = (self.get('Subtitles', 'format', 'srt') or 'srt').strip().lower()
        return v if v in ('srt', 'ass') else 'srt'

    @property
    def subtitle_fix_linger(self):
        """是否自动修复字幕滞留（压缩异常过长的结束时间）"""
        return self.get_bool('Subtitles', 'fix_linger', True)

    @property
    def subtitle_min_duration_sec(self):
        """字幕最短显示时长（秒）"""
        return self.get_float('Subtitles', 'min_duration_sec', 1.2)

    @property
    def subtitle_max_duration_sec(self):
        """字幕最长显示时长（秒），用于避免长静音导致的“字幕滞留”"""
        return self.get_float('Subtitles', 'max_duration_sec', 20.0)

    @property
    def subtitle_chars_per_sec(self):
        """估算阅读速度：每秒字符数（中文建议 8~12）"""
        return self.get_float('Subtitles', 'chars_per_sec', 5.0)

    @property
    def subtitle_linger_slack_sec(self):
        """超过“可读时长 + slack”才收缩 end（秒），避免过度干预"""
        return self.get_float('Subtitles', 'linger_slack_sec', 0.8)

    @property
    def subtitle_linger_trigger_sec(self):
        """仅当段落时长 >= trigger_sec 才考虑收缩（秒）"""
        return self.get_float('Subtitles', 'linger_trigger_sec', 6.0)

    @property
    def subtitle_linger_trigger_ratio(self):
        """仅当段落时长 > 可读时长 * trigger_ratio 才收缩（更保守，避免提前结束）"""
        return self.get_float('Subtitles', 'linger_trigger_ratio', 6.0)

    @property
    def subtitle_linger_keep_ratio(self):
        """触发收缩后，保留的时长=可读时长*keep_ratio（避免太短）"""
        return self.get_float('Subtitles', 'linger_keep_ratio', 4.0)

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

    @property
    def vocal_separation_chunk_sec(self):
        """Demucs 分段秒数（用于超长音频避免 OOM；默认 1800=30分钟）"""
        return self.get_int('Audio', 'vocal_separation_chunk_sec', 1800)


# 全局配置实例
config = Config()

if __name__ == '__main__':
    # 测试配置
    print("配置测试：")
    print(f"DeepSeek API密钥: {'已设置' if config.deepseek_api_key else '未设置'}")
    print(f"服务地址: {config.service_host}:{config.service_port}")
    print(f"服务端按需加载模型: {config.lazy_load_models}")
    print(f"客户端按需管理模型: {config.manage_models}")
    print(f"任务后卸载模型: {config.unload_models_after_tasks}")
    print(f"ASR模型: {config.asr_model_size}")
    print(f"翻译模型: {config.translation_model}")
    print(f"使用GPU: {config.use_gpu}")
    print(f"beam_size: {config.beam_size}")
    print(f"GPU任务前清理显存缓存: {config.clear_cuda_cache_before_tasks}")
    print(f"ASR语言: {config.asr_language}")
    print(f"ASR分块秒数: {config.asr_chunk_sec}")
    print(f"ASR分块重叠秒数: {config.asr_chunk_overlap_sec}")
    print(f"默认目标语言: {config.default_target_language}")
    print(f"使用DeepSeek润色: {config.use_deepseek_polish}")
    print(f"启用人声分离: {config.enable_vocal_separation}")
    print(f"人声分离模型: {config.vocal_separation_model}")
    print(f"人声分离设备: {config.vocal_separation_device}")
