#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量视频翻译工具 v3.1
支持：上下文翻译、并发DeepSeek润色、断点续传、日志记录
"""

import os
import sys
import time
import json
import argparse
import logging
import threading
import tempfile
import re
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import subprocess
import shutil

# 解决Windows终端编码问题
if sys.platform == 'win32':
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 导入配置管理器
try:
    from config_manager import config

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("警告: 未找到config_manager.py，将仅使用环境变量或命令行参数")


def setup_logger():
    """配置日志系统"""
    log_dir = Path('log')
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'translation_{timestamp}.log'

    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_file


def cleanup_old_logs(log_dir, keep_days=7, auto_cleanup=False):
    """
    清理旧日志文件

    Args:
        log_dir: 日志目录路径
        keep_days: 保留最近几天的日志（默认7天）
        auto_cleanup: 是否自动清理（默认False，需要配置启用）

    Returns:
        int: 删除的文件数量
    """
    if not auto_cleanup:
        return 0

    log_dir = Path(log_dir)
    if not log_dir.exists():
        return 0

    # 获取当前时间
    now = time.time()
    cutoff_time = now - (keep_days * 24 * 3600)

    deleted_count = 0
    deleted_size = 0

    # 遍历日志目录
    for log_file in log_dir.glob('translation_*.log'):
        try:
            # 获取文件修改时间
            file_mtime = log_file.stat().st_mtime

            # 如果文件超过保留期限
            if file_mtime < cutoff_time:
                file_size = log_file.stat().st_size
                log_file.unlink()
                deleted_count += 1
                deleted_size += file_size
        except Exception as e:
            # 删除失败时忽略，继续处理其他文件
            pass

    if deleted_count > 0:
        size_mb = deleted_size / (1024 * 1024)
        print(f"🗑️  已清理 {deleted_count} 个超过 {keep_days} 天的旧日志文件（释放 {size_mb:.1f}MB）")

    return deleted_count


class VideoTranslator:
    """视频批量翻译器（支持并发润色）"""

    def __init__(self, service_url='http://127.0.0.1:50515', deepseek_key=None,
                 use_polish=False, concurrent_polish=10,
                 enable_vocal_separation=False, vocal_separation_model='htdemucs', vocal_separation_device='auto',
                 vocal_separation_chunk_sec=1800,
                 clear_cuda_cache_before_tasks=False,
                 asr_chunk_sec=0, asr_chunk_overlap_sec=0.0,
                 manage_models=False, unload_models_after_tasks=False, model_load_timeout=3600,
                 subtitle_format: str = 'srt'):
        self.service_url = service_url

        # 优先级：命令行参数 > 环境变量 > config.ini
        if deepseek_key:
            self.deepseek_key = deepseek_key
        elif os.getenv('DEEPSEEK_API_KEY'):
            self.deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        elif CONFIG_AVAILABLE:
            self.deepseek_key = config.deepseek_api_key
        else:
            self.deepseek_key = None

        self.use_polish = use_polish and self.deepseek_key
        self.concurrent_polish = concurrent_polish  # 并发数

        # 可选：人声分离（Demucs）用于提升嘈杂/背景音乐场景识别效果
        self.enable_vocal_separation = enable_vocal_separation
        self.vocal_separation_model = vocal_separation_model
        self.vocal_separation_device = (vocal_separation_device or 'auto').lower()
        try:
            self.vocal_separation_chunk_sec = int(vocal_separation_chunk_sec)
        except Exception:
            self.vocal_separation_chunk_sec = 1800
        if self.vocal_separation_chunk_sec <= 0:
            self.vocal_separation_chunk_sec = 1800
        self.clear_cuda_cache_before_tasks = bool(clear_cuda_cache_before_tasks)
        self.asr_chunk_sec = int(asr_chunk_sec or 0)
        self.asr_chunk_overlap_sec = float(asr_chunk_overlap_sec or 0.0)
        self.manage_models = bool(manage_models)
        self.unload_models_after_tasks = bool(unload_models_after_tasks)
        self.model_load_timeout = int(model_load_timeout or 3600)
        self.subtitle_format = (subtitle_format or 'srt').strip().lower()
        if self.subtitle_format not in ('srt', 'ass'):
            self.subtitle_format = 'srt'

        # 线程池用于并发润色
        if self.use_polish:
            self.polish_executor = ThreadPoolExecutor(max_workers=concurrent_polish)
            self.polish_lock = threading.Lock()  # 用于日志同步

        # 统计信息
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None
        }

        # 支持的视频格式
        self.video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm', '.m4v'}

        # 进度管理
        self.progress_dir = Path('.progress')
        self.progress_dir.mkdir(exist_ok=True)
        self.progress_file = None
        self.progress_data = {}

    def __del__(self):
        """清理线程池"""
        if hasattr(self, 'polish_executor'):
            self.polish_executor.shutdown(wait=True)

    def clear_cuda_cache(self, stage: str):
        """清理 CUDA 显存缓存（尽量减少 OOM 概率）"""
        if not self.clear_cuda_cache_before_tasks:
            return

        # 1) 当前进程（若安装了 torch）
        try:
            import gc

            gc.collect()
        except Exception:
            pass

        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

        # 2) 服务端进程（如果提供了 /gpu/clear）
        try:
            response = requests.post(
                f"{self.service_url}/gpu/clear",
                json={'stage': stage},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json() or {}
                if data.get('cuda_available') and data.get('free_bytes') and data.get('total_bytes'):
                    free_gb = data['free_bytes'] / (1024 ** 3)
                    total_gb = data['total_bytes'] / (1024 ** 3)
                    logging.info(f"    [GPU] 已清理缓存: {stage}（free {free_gb:.1f}GB / total {total_gb:.1f}GB）")
        except Exception:
            # 不影响主流程
            pass

    def _service_models_load(self, models: list[str]) -> bool:
        """让服务端按需加载指定模型（需要 server_optimized.py 支持 /models/load）。"""
        try:
            resp = requests.post(
                f"{self.service_url}/models/load",
                json={'models': models},
                timeout=30
            )
        except Exception as e:
            logging.error(f"  × 请求服务端加载模型失败: {e}")
            return False

        if resp.status_code == 404:
            logging.error("  × 服务端不支持 /models/load：请更新 server_optimized.py")
            return False

        if resp.status_code != 200:
            try:
                logging.error(f"  × 服务端加载模型失败: {resp.status_code} {resp.json()}")
            except Exception:
                logging.error(f"  × 服务端加载模型失败: {resp.status_code} {resp.text[:500]}")
            return False

        return bool((resp.json() or {}).get('success', True))

    def _service_models_unload(self, models: list[str]) -> bool:
        """让服务端卸载指定模型以释放显存（需要 /models/unload）。"""
        try:
            resp = requests.post(
                f"{self.service_url}/models/unload",
                json={'models': models},
                timeout=30
            )
        except Exception:
            return False

        if resp.status_code != 200:
            return False
        return True

    def _wait_models_ready(self, want_asr: bool, want_translation: bool, timeout: int) -> bool:
        """等待服务端指定模型就绪。"""
        start = time.time()
        last_log = 0.0
        while True:
            if time.time() - start > timeout:
                logging.error(f"  × 等待模型加载超时（{timeout}秒）")
                return False
            try:
                # 模型首次下载/加载时，服务端可能短暂无响应（GIL/IO/CPU占用），适当放宽 read timeout
                h = requests.get(f"{self.service_url}/health", timeout=(3, 30)).json()
            except Exception as e:
                if time.time() - last_log >= 10:
                    logging.info(f"    … 等待服务响应: {e}")
                    last_log = time.time()
                time.sleep(2)
                continue

            if h.get('phase') == 'error':
                logging.error(f"  × 模型加载失败: {h.get('error') or '未知错误'}")
                return False

            asr_ready = bool(h.get('asr_ready'))
            translation_ready = bool(h.get('translation_ready'))
            if (not want_asr or asr_ready) and (not want_translation or translation_ready):
                return True

            now = time.time()
            if now - last_log >= 10:
                phase = h.get('phase')
                pct = int((h.get('progress') or 0) * 100)
                msg = h.get('message') or ''
                logging.info(f"    … 模型加载中: {phase} ({pct}%) {msg}".rstrip())
                last_log = now
            time.sleep(2)

    def ensure_models(self, want_asr: bool = False, want_translation: bool = False) -> bool:
        """按需加载服务端模型，并等待就绪。"""
        models: list[str] = []
        if want_asr:
            models.append('asr')
        if want_translation:
            models.append('translation')
        if not models:
            return True

        if not self._service_models_load(models):
            return False
        return self._wait_models_ready(want_asr, want_translation, timeout=self.model_load_timeout)

    def load_progress(self, task_name):
        """加载进度文件"""
        self.progress_file = self.progress_dir / f'{task_name}.json'

        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    self.progress_data = json.load(f)
                logging.info(f"✓ 加载进度文件: {self.progress_file.name}")

                completed = sum(1 for v in self.progress_data.values() if v.get('status') == 'completed')
                failed = sum(1 for v in self.progress_data.values() if v.get('status') == 'failed')
                if completed > 0 or failed > 0:
                    logging.info(f"  已完成: {completed}, 已失败: {failed}")
            except:
                self.progress_data = {}
        else:
            self.progress_data = {}

    def save_progress(self):
        """保存进度"""
        if self.progress_file:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, ensure_ascii=False, indent=2)

    def update_video_status(self, video_name, status, **kwargs):
        """更新视频状态"""
        self.progress_data[video_name] = {
            'status': status,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **kwargs
        }
        self.save_progress()

    def should_skip_video(self, video_path, srt_path):
        """检查是否应该跳过该视频"""
        video_name = video_path.name

        # 检查字幕文件是否存在
        if srt_path.exists():
            return True, '字幕文件已存在'

        if video_name not in self.progress_data:
            return False, None

        status = self.progress_data[video_name].get('status')

        if status == 'completed':
            return False, '上次完成但文件缺失，重新处理'
        elif status == 'processing':
            return False, '上次未完成，重新处理'
        elif status == 'failed':
            retry_count = self.progress_data[video_name].get('retry_count', 0)
            if retry_count >= 3:
                return True, f'已失败{retry_count}次，跳过'
            else:
                return False, f'重试第{retry_count + 1}次'

        return False, None

    def check_service(self, wait_ready=False, wait_timeout=3600, poll_interval=2):
        """检查翻译服务是否可用（可选等待服务就绪）"""
        start_time = time.time()
        last_status = None
        last_log_time = 0.0
        last_error_log_time = 0.0

        while True:
            try:
                response = requests.get(f"{self.service_url}/health", timeout=5)
                if response.status_code != 200:
                    logging.error(f"× 翻译服务异常: {response.status_code}")
                    return False

                data = response.json()

                if data.get('ready'):
                    logging.info("✓ 翻译服务正常运行")
                    return True

                if not wait_ready:
                    if self.manage_models:
                        logging.info("✓ 翻译服务正常运行（模型将按需加载）")
                        return True
                    logging.error("× 翻译服务未就绪，请等待模型加载")
                    return False

                phase = data.get('phase')
                progress = data.get('progress')
                message = data.get('message')
                error = data.get('error')

                # 仅在状态变化时输出，避免刷屏
                status = (phase, progress, message, error)
                now = time.time()
                if status != last_status or (now - last_log_time) >= 10:
                    pct = int((progress or 0) * 100)
                    logging.info(f"… 等待服务就绪: {phase} ({pct}%) {message or ''}".rstrip())
                    last_status = status
                    last_log_time = now

                if phase == 'error':
                    logging.error(f"× 模型加载失败: {error or '未知错误'}")
                    return False

            except Exception as e:
                if not wait_ready:
                    logging.error(f"× 无法连接到翻译服务: {e}")
                    logging.error("  请确保服务正在运行: python server_optimized.py")
                    return False
                now = time.time()
                if (now - last_error_log_time) >= 10:
                    logging.info(f"… 等待翻译服务启动: {e}")
                    last_error_log_time = now

            if time.time() - start_time > wait_timeout:
                logging.error(f"× 等待翻译服务就绪超时（{wait_timeout}秒）")
                logging.error("  你可以：1) 继续等待并重试 2) 查看 server.log 3) 换更小的翻译模型")
                return False

            time.sleep(poll_interval)

    def extract_audio(self, video_path, output_path, sample_rate=16000, channels=1):
        """从视频提取音频"""
        try:
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(sample_rate), '-ac', str(channels),
                '-y', output_path
            ]

            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"  × 音频提取失败: {e}")
            return False
        except FileNotFoundError:
            logging.error("  × 找不到ffmpeg，请安装ffmpeg并添加到PATH")
            return False

    def separate_vocals(self, input_audio_path: Path, output_wav_path: Path):
        """
        使用 Demucs 做人声分离，输出 16kHz/mono WAV 供 ASR 使用。

        需要额外安装：
          pip install demucs
        """
        try:
            import demucs  # noqa: F401
        except Exception as e:
            raise RuntimeError("已启用人声分离，但未安装 demucs：请先运行 `pip install demucs`") from e

        def _ffprobe_duration_sec(path: Path) -> float | None:
            try:
                p = subprocess.run(
                    [
                        'ffprobe',
                        '-v', 'error',
                        '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1',
                        str(path),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if p.returncode != 0:
                    return None
                s = (p.stdout or '').strip()
                if not s:
                    return None
                return float(s)
            except Exception:
                return None

        # 通过环境变量控制 Demucs 是否使用 GPU（避免依赖不同版本的 CLI 参数）
        env = os.environ.copy()
        if self.vocal_separation_device == 'cpu':
            env['CUDA_VISIBLE_DEVICES'] = ''
        elif self.vocal_separation_device == 'cuda':
            # 尽量减少显存碎片导致的 OOM（不覆盖用户已有设置）
            # 注：PYTORCH_CUDA_ALLOC_CONF 已弃用，改用 PYTORCH_ALLOC_CONF
            env.setdefault('PYTORCH_ALLOC_CONF', 'max_split_size_mb:128')

        def _run_demucs_one(track_path: Path, out_16k_wav_path: Path, label: str | None = None):
            tmp_dir = tempfile.mkdtemp(prefix='demucs_')
            success = False
            try:
                cmd = [
                    sys.executable, '-m', 'demucs.separate',
                    '-n', self.vocal_separation_model,
                    '--two-stems', 'vocals',
                    '-o', tmp_dir,
                    str(track_path)
                ]
                demucs_log_path = Path(tmp_dir) / 'demucs.log'
                env.setdefault('PYTHONUNBUFFERED', '1')

                is_tty = sys.stdout.isatty()
                percent_re = re.compile(r'(\d{1,3})%')
                last_percent = None
                next_log_percent = 10
                spinner = ['|', '/', '-', '\\']
                spinner_i = 0
                start_time = time.time()
                log_offset = 0
                log_buf = ''

                prefix = "    [Demucs]" if not label else f"    [Demucs {label}]"

                def _print_progress(text: str):
                    if not is_tty:
                        return
                    sys.stdout.write(text)
                    sys.stdout.flush()

                with open(demucs_log_path, 'wb') as demucs_log:
                    proc = subprocess.Popen(
                        cmd,
                        env=env,
                        stdout=demucs_log,
                        stderr=subprocess.STDOUT,
                    )

                    while True:
                        rc = proc.poll()

                        # 尝试从 demucs.log 读取新增内容并解析百分比
                        try:
                            with open(demucs_log_path, 'rb') as f:
                                f.seek(log_offset, os.SEEK_SET)
                                data = f.read()
                                log_offset = f.tell()
                            if data:
                                chunk = data.decode('utf-8', errors='replace')
                                log_buf = (log_buf + chunk)[-50_000:]  # 仅保留末尾，避免无限增长
                                matches = percent_re.findall(log_buf)
                                if matches:
                                    p = int(matches[-1])
                                    if 0 <= p <= 100:
                                        if last_percent is None:
                                            last_percent = p
                                        else:
                                            # tqdm 可能会有多段进度条，允许在接近完成后重置
                                            if p < last_percent and last_percent >= 95 and p <= 5:
                                                last_percent = p
                                                next_log_percent = 10
                                            else:
                                                last_percent = p
                        except Exception:
                            pass

                        elapsed = int(time.time() - start_time)
                        if last_percent is not None:
                            bar_len = 24
                            filled = int(bar_len * last_percent / 100)
                            bar = '#' * filled + '.' * (bar_len - filled)
                            _print_progress(f"\r{prefix} {last_percent:3d}% |{bar}| {elapsed}s")

                            # notebook/非TTY：每10%记录一次，避免刷屏
                            if (not is_tty) and last_percent >= next_log_percent:
                                logging.info(f"{prefix} 进度: {last_percent}%")
                                next_log_percent += 10
                        else:
                            _print_progress(f"\r{prefix} {spinner[spinner_i % len(spinner)]} 运行中... {elapsed}s")
                            spinner_i += 1

                        if rc is not None:
                            break
                        time.sleep(0.5)

                    proc.wait()

                if is_tty:
                    # 清理进度行
                    sys.stdout.write("\n")
                    sys.stdout.flush()

                if proc.returncode != 0:
                    # 避免 stdout PIPE 卡死：输出写入文件，失败时仅打印末尾
                    tail_text = ''
                    try:
                        with open(demucs_log_path, 'rb') as f:
                            f.seek(0, os.SEEK_END)
                            size = f.tell()
                            f.seek(max(0, size - 200_000), os.SEEK_SET)  # 只读末尾200KB
                            data = f.read()
                        tail_text = data.decode('utf-8', errors='replace').strip()
                    except Exception:
                        tail_text = ''
                    if tail_text:
                        tail = "\n".join(tail_text.splitlines()[-200:])
                        logging.error("  × Demucs 输出（最后200行）：\n" + tail)
                    logging.error(f"  ! Demucs 日志文件: {demucs_log_path}")
                    if 'TorchCodec is required' in tail_text or "No module named 'torchcodec'" in tail_text:
                        raise RuntimeError("Demucs 依赖 torchcodec 保存音频：请先运行 `pip install torchcodec` 再重试") from None
                    if proc.returncode == -9:
                        raise RuntimeError("Demucs 被系统终止（exit code=-9）：通常是 CPU 内存不足（OOM kill）。建议启用分段人声分离或缩短分段时长。") from None
                    raise RuntimeError(f"Demucs 执行失败（exit code={proc.returncode}）: {' '.join(cmd)}")

                tmp_dir_path = Path(tmp_dir)
                candidates = list(tmp_dir_path.rglob('vocals.wav'))
                if not candidates:
                    candidates = [p for p in tmp_dir_path.rglob('vocals.*') if p.is_file()]
                if not candidates:
                    raise FileNotFoundError(f"Demucs 输出未找到：{tmp_dir}")

                vocals_path = candidates[0]
                cmd = [
                    'ffmpeg', '-i', str(vocals_path),
                    '-vn', '-acodec', 'pcm_s16le',
                    '-ar', '16000', '-ac', '1',
                    '-y', str(out_16k_wav_path)
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if not out_16k_wav_path.exists():
                    raise FileNotFoundError(f"人声分离输出未生成: {out_16k_wav_path}")
                success = True
            finally:
                if success:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                else:
                    logging.error(f"  ! Demucs 临时目录已保留用于排查: {tmp_dir}")

        # 超长音频：避免 demucs/torchaudio 一次性加载到内存导致 OOM kill（exit code=-9）
        size_bytes = None
        try:
            size_bytes = input_audio_path.stat().st_size
        except Exception:
            size_bytes = None
        duration_sec = _ffprobe_duration_sec(input_audio_path)

        # 触发条件：>1小时 或 >512MB（约 48min 的 44.1kHz/2ch/16bit PCM）
        enable_chunk = True
        chunk_sec = int(getattr(self, 'vocal_separation_chunk_sec', 1800) or 1800)
        chunk_threshold_sec = 3600
        size_threshold_bytes = 512 * 1024 * 1024
        should_chunk = (
            enable_chunk
            and chunk_sec > 0
            and (
                (duration_sec is not None and duration_sec >= chunk_threshold_sec)
                or (size_bytes is not None and size_bytes >= size_threshold_bytes)
            )
        )

        if not should_chunk:
            _run_demucs_one(input_audio_path, output_wav_path)
            return

        work_dir = tempfile.mkdtemp(prefix='demucs_chunks_')
        work_path = Path(work_dir)
        success = False
        try:
            if duration_sec is not None:
                logging.info(f"    [Demucs] 检测到长音频（{duration_sec/60:.1f}分钟），启用分段处理：{chunk_sec}s/段")
            elif size_bytes is not None:
                logging.info(f"    [Demucs] 检测到大音频（{size_bytes/(1024**3):.2f}GB），启用分段处理：{chunk_sec}s/段")
            else:
                logging.info(f"    [Demucs] 启用分段处理：{chunk_sec}s/段")

            chunk_pattern = work_path / 'chunk_%06d.wav'
            cmd = [
                'ffmpeg',
                '-i', str(input_audio_path),
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '44100', '-ac', '2',
                '-f', 'segment',
                '-segment_time', str(int(chunk_sec)),
                '-reset_timestamps', '1',
                '-y', str(chunk_pattern),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            chunks = sorted(work_path.glob('chunk_*.wav'))
            if not chunks:
                raise RuntimeError("分段失败：未生成任何音频分段文件")

            # 分段生成后即可删除原始大文件，避免占用磁盘（若失败可从视频重新提取）
            try:
                if size_bytes is not None and size_bytes >= size_threshold_bytes and input_audio_path.exists():
                    input_audio_path.unlink(missing_ok=True)
            except Exception:
                pass

            chunk_vocals = []
            total = len(chunks)
            for i, chunk_path in enumerate(chunks, 1):
                label = f"{i}/{total}"
                out_chunk = work_path / f"vocals_{i:06d}.wav"
                _run_demucs_one(chunk_path, out_chunk, label=label)
                chunk_vocals.append(out_chunk)

            concat_list = work_path / 'concat.txt'
            with open(concat_list, 'w', encoding='utf-8') as f:
                for p in chunk_vocals:
                    pp = str(p).replace("'", "'\\''")
                    f.write(f"file '{pp}'\n")

            # 优先用 -c copy（同一编码参数的 WAV 可直接拼接），失败则回退到重编码
            concat_cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_list),
                '-c', 'copy',
                '-y', str(output_wav_path),
            ]
            try:
                subprocess.run(concat_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                concat_cmd = [
                    'ffmpeg',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', str(concat_list),
                    '-vn',
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000', '-ac', '1',
                    '-y', str(output_wav_path),
                ]
                subprocess.run(concat_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if not output_wav_path.exists():
                raise FileNotFoundError(f"人声分离输出未生成: {output_wav_path}")

            success = True
        finally:
            if success:
                shutil.rmtree(work_dir, ignore_errors=True)
            else:
                logging.error(f"  ! Demucs 分段临时目录已保留用于排查: {work_dir}")

    @staticmethod
    def _trim_text_overlap(prev_text: str, text: str, max_words: int = 12) -> str:
        """
        处理分块识别时的边界重复：若上一段尾部若干单词与本段开头重复，则从本段移除该重复部分。
        """
        prev_text = (prev_text or '').strip()
        text = (text or '').strip()
        if not prev_text or not text:
            return text

        prev_words = prev_text.split()
        words = text.split()
        if not prev_words or not words:
            return text

        def _norm_word(w: str) -> str:
            return re.sub(r'^[\\W_]+|[\\W_]+$', '', w).lower()

        prev_norm = [_norm_word(w) for w in prev_words]
        norm = [_norm_word(w) for w in words]
        max_k = min(max_words, len(prev_norm), len(norm))
        for k in range(max_k, 0, -1):
            if prev_norm[-k:] == norm[:k] and all(prev_norm[-k:]) and all(norm[:k]):
                return " ".join(words[k:]).lstrip()
        return text

    def _transcribe_http(self, audio_file, filename: str, language: str | None, timeout: int = 3600):
        """调用服务端 /transcribe（audio_file 为文件对象或 BytesIO）。"""
        files = {'audio': (filename, audio_file)}
        data = {}
        if language and language != 'auto':
            data['language'] = language

        response = requests.post(
            f"{self.service_url}/transcribe",
            files=files,
            data=data,
            timeout=timeout
        )

        if response.status_code == 200:
            return response.json()

        logging.error(f"\n  × 识别失败: {response.status_code}")
        # 尽量打印服务端返回的错误信息，方便在无 server.log 的环境排查
        try:
            data = response.json()
            if isinstance(data, dict):
                err = data.get('error') or data
            else:
                err = data
            logging.error(f"    服务端错误: {err}")
            if isinstance(data, dict) and data.get('traceback'):
                logging.error("    服务端 traceback（末尾）：\n" + str(data.get('traceback'))[-2000:])
        except Exception:
            body = (response.text or '').strip()
            if body:
                body = body[:2000]
                logging.error(f"    服务端响应: {body}")
        return None

    def transcribe_single(self, audio_path: str, language: str | None = None):
        """语音识别（整段，阻塞等待服务端返回）。"""
        try:
            with open(audio_path, 'rb') as f:
                return self._transcribe_http(f, Path(audio_path).name, language, timeout=3600)
        except requests.exceptions.Timeout:
            logging.error("\n  × 识别超时（视频太长，超过1小时处理时间）")
            return None
        except Exception as e:
            logging.error(f"\n  × 识别错误: {e}")
            return None

    def transcribe_chunked(self, audio_path: str, chunk_sec: int, overlap_sec: float = 0.0, language: str | None = None):
        """语音识别（分块上传，显示进度；也可降低长音频导致的 500/OOM 概率）。"""
        import io
        import math
        import wave

        try:
            chunk_sec = int(chunk_sec)
            overlap_sec = float(overlap_sec or 0.0)
        except Exception:
            return self.transcribe_single(audio_path, language=language)

        if chunk_sec <= 0:
            return self.transcribe_single(audio_path, language=language)
        if overlap_sec < 0:
            overlap_sec = 0.0
        if overlap_sec >= chunk_sec:
            overlap_sec = max(0.0, chunk_sec - 0.1)

        try:
            with wave.open(audio_path, 'rb') as wf:
                nchannels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                nframes = wf.getnframes()

                if framerate <= 0 or nframes <= 0:
                    return self.transcribe_single(audio_path, language=language)

                total_sec = nframes / float(framerate)
                chunk_frames = max(1, int(chunk_sec * framerate))
                overlap_frames = max(0, int(overlap_sec * framerate))
                step_frames = max(1, chunk_frames - overlap_frames)

                if nframes <= chunk_frames:
                    return self.transcribe_single(audio_path, language=language)

                total_chunks = int(math.ceil((nframes - chunk_frames) / step_frames)) + 1

                is_tty = sys.stdout.isatty()
                bar_len = 24
                start_time = time.time()
                last_percent = -1

                def _print_progress(text: str):
                    if not is_tty:
                        return
                    sys.stdout.write(text)
                    sys.stdout.flush()

                merged_segments: list[dict] = []
                merged_text = ''
                detected_language = None
                detected_prob = None
                total_processing_ms = 0

                start_frame = 0
                chunk_index = 0
                while start_frame < nframes:
                    end_frame = min(nframes, start_frame + chunk_frames)
                    chunk_start_sec = start_frame / float(framerate)
                    chunk_end_sec = end_frame / float(framerate)

                    chunk_index += 1

                    wf.setpos(start_frame)
                    frames = wf.readframes(end_frame - start_frame)
                    bio = io.BytesIO()
                    with wave.open(bio, 'wb') as out_wav:
                        out_wav.setnchannels(nchannels)
                        out_wav.setsampwidth(sampwidth)
                        out_wav.setframerate(framerate)
                        out_wav.writeframes(frames)
                    bio.seek(0)

                    lang_to_use = None
                    if language and language != 'auto':
                        lang_to_use = language
                    elif detected_language:
                        lang_to_use = detected_language

                    result = self._transcribe_http(
                        bio,
                        filename=f"chunk_{chunk_index}.wav",
                        language=lang_to_use,
                        timeout=3600
                    )
                    if not result or not result.get('success'):
                        if is_tty:
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                        return None

                    if detected_language is None:
                        detected_language = result.get('language')
                        detected_prob = result.get('language_probability')
                    try:
                        total_processing_ms += int(result.get('processing_time_ms') or 0)
                    except Exception:
                        pass

                    for seg in result.get('segments', []) or []:
                        try:
                            seg_start = float(seg.get('start', 0.0)) + chunk_start_sec
                            seg_end = float(seg.get('end', 0.0)) + chunk_start_sec
                        except Exception:
                            continue
                        seg_text_raw = seg.get('text', '')
                        if not str(seg_text_raw).strip():
                            continue

                        if merged_segments:
                            last = merged_segments[-1]
                            last_end = float(last.get('end', 0.0))
                            if seg_end <= last_end + 0.02:
                                continue
                            if seg_start < last_end - 0.02:
                                trimmed = self._trim_text_overlap(last.get('text', ''), str(seg_text_raw))
                                if trimmed and trimmed != str(seg_text_raw).strip():
                                    seg_text_raw = trimmed
                                seg_start = max(seg_start, last_end)

                        if seg_end <= seg_start + 0.02:
                            continue

                        merged_segments.append({'start': seg_start, 'end': seg_end, 'text': seg_text_raw})
                        try:
                            merged_text += str(seg_text_raw)
                        except Exception:
                            pass

                    elapsed = int(time.time() - start_time)
                    percent = int(min(100.0, (chunk_end_sec / total_sec) * 100.0))
                    if is_tty and percent != last_percent:
                        filled = int(bar_len * percent / 100)
                        bar = '#' * filled + '.' * (bar_len - filled)
                        _print_progress(f"\r    [ASR] {percent:3d}% |{bar}| {elapsed}s (chunk {chunk_index}/{total_chunks})")
                        last_percent = percent
                    elif not is_tty:
                        filled = int(bar_len * percent / 100)
                        bar = '#' * filled + '.' * (bar_len - filled)
                        logging.info(f"    [ASR] {percent:3d}% |{bar}| {elapsed}s (chunk {chunk_index}/{total_chunks})")

                    if end_frame >= nframes:
                        break
                    start_frame += step_frames

                if is_tty:
                    sys.stdout.write("\n")
                    sys.stdout.flush()

                return {
                    'success': True,
                    'text': (merged_text or '').strip(),
                    'language': detected_language,
                    'language_probability': detected_prob,
                    'segments': merged_segments,
                    'processing_time_ms': total_processing_ms,
                }
        except wave.Error:
            return self.transcribe_single(audio_path, language=language)
        except Exception as e:
            logging.error(f"\n  × 分块识别错误: {e}")
            return self.transcribe_single(audio_path, language=language)

    def transcribe(self, audio_path: str, language: str | None = None):
        """语音识别（支持分块进度条）。"""
        if self.asr_chunk_sec and self.asr_chunk_sec > 0:
            return self.transcribe_chunked(
                audio_path,
                chunk_sec=self.asr_chunk_sec,
                overlap_sec=self.asr_chunk_overlap_sec,
                language=language
            )
        return self.transcribe_single(audio_path, language=language)

    def translate_text(self, text, source_lang='en', target_lang='zh', max_retries=3):
        """翻译文本（带重试）"""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.service_url}/translate",
                    json={
                        'text': text,
                        'source_language': source_lang,
                        'target_language': target_lang
                    },
                    timeout=90
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get('translated_text', '')
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return text
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    logging.warning(f"  ! 翻译超时")
                    return text
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    logging.warning(f"  ! 翻译失败: {e}")
                    return text
        return text

    def get_context_window(self, segments, index, window_size=2):
        """获取上下文窗口"""
        start = max(0, index - window_size)
        end = min(len(segments), index + window_size + 1)

        context_before = []
        for i in range(start, index):
            if 'translated' in segments[i]:
                context_before.append(segments[i]['translated'])

        context_after = []
        for i in range(index + 1, end):
            context_after.append(segments[i]['text'])

        return context_before, context_after

    def polish_translation_with_context(self, text, translated, context_before, context_after,
                                        source_lang='en', target_lang='zh', max_retries=3):
        """使用DeepSeek润色翻译（带上下文）"""
        if not self.use_polish:
            return translated

        lang_names = {'en': '英语', 'zh': '中文', 'ja': '日语', 'ko': '韩语'}
        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)

        # 构建带上下文的提示词
        context_str = ""
        if context_before:
            context_str += f"\n前文（已翻译）：\n" + "\n".join(f"- {c}" for c in context_before[-2:])

        if context_after:
            context_str += f"\n\n后文（原文）：\n" + "\n".join(f"- {c}" for c in context_after[:2])

        prompt = f"""你是专业的{target_name}影视字幕翻译专家。请结合上下文，将以下{source_name}对话翻译得更地道、自然。
{context_str}

当前句子：
原文：{text}
机器翻译：{translated}

润色要求：
1. 结合上下文理解对话情境和人物关系
2. 准确传达原意、语气和情感
3. 使用最自然地道的{target_name}口语表达
4. 避免书面语和直译腔
5. 保持与上下文的连贯性
6. **重要：只返回这一句话的润色翻译，不要分成多行，不要添加其他句子**
7. 不要任何解释、标点符号或多余内容

润色后："""

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    'https://api.deepseek.com/v1/chat/completions',
                    headers={
                        'Authorization': f'Bearer {self.deepseek_key}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': 'deepseek-chat',
                        'messages': [
                            {'role': 'system', 'content': f'你是专业的{target_name}影视字幕翻译专家。'},
                            {'role': 'user', 'content': prompt}
                        ],
                        'temperature': 0.5,
                        'max_tokens': 500
                    },
                    timeout=90
                )

                if response.status_code == 200:
                    result = response.json()
                    polished = result['choices'][0]['message']['content'].strip()

                    # 如果返回多行，只取第一行（修复DeepSeek可能返回多行的问题）
                    if '\n' in polished:
                        polished = polished.split('\n')[0].strip()

                    # 清理可能的多余字符（如开头的"- "等）
                    polished = polished.lstrip('- •·').strip()

                    # 清理引号
                    polished = polished.strip('"\'').strip()

                    # 最终验证：如果结果为空或太短，使用原译文
                    if not polished or len(polished) < 2:
                        return translated

                    return polished
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return translated
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return translated
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return translated
        return translated

    def polish_batch_with_context(self, segments, source_lang, target_lang, step_idx=None, step_total=None):
        """批量并发润色（带上下文）"""
        if not self.use_polish:
            return

        if step_idx and step_total:
            logging.info(f"  [{step_idx}/{step_total}] DeepSeek并发润色（{self.concurrent_polish}线程）...")
        else:
            logging.info(f"  DeepSeek并发润色（{self.concurrent_polish}线程）...")

        # 提交所有任务
        futures = {}
        for i, seg in enumerate(segments):
            context_before, context_after = self.get_context_window(segments, i, window_size=2)

            future = self.polish_executor.submit(
                self.polish_translation_with_context,
                seg['text'],
                seg['translated'],
                context_before,
                context_after,
                source_lang,
                target_lang
            )
            futures[future] = i

        # 收集结果
        completed = 0
        polish_examples = []  # 记录润色示例
        total = len(segments)

        for future in as_completed(futures):
            i = futures[future]
            try:
                polished = future.result(timeout=120)  # 2分钟超时

                # 记录变化（前3个示例）
                if polished != segments[i]['translated'] and len(polish_examples) < 3:
                    context_before, context_after = self.get_context_window(segments, i, 2)
                    polish_examples.append({
                        'index': i + 1,
                        'original': segments[i]['translated'],
                        'polished': polished,
                        'context_before': context_before,
                        'context_after': context_after
                    })

                segments[i]['translated'] = polished

            except Exception as e:
                # 失败时保持原译文
                pass

            completed += 1
            # 每完成20%显示一次进度
            if completed % max(1, total // 5) == 0 or completed == total:
                logging.info(f"    润色进度: {completed}/{total} ({completed * 100 // total}%)")

        if step_idx and step_total:
            logging.info(f"  [{step_idx}/{step_total}] 并发润色完成 ✓")
        else:
            logging.info("  并发润色完成 ✓")

        # 显示润色示例
        if polish_examples:
            logging.info("")
            for example in polish_examples:
                if example['context_before']:
                    logging.info(f"    上文: ...{example['context_before'][-1]}")
                logging.info(f"    [{example['index']}] 原译: {example['original']}")
                logging.info(f"    [{example['index']}] 润色: {example['polished']}")
                if example['context_after']:
                    logging.info(f"    下文: {example['context_after'][0]}...")
                logging.info("")

    def format_time(self, seconds):
        """格式化时间为SRT格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def format_time_ass(seconds: float) -> str:
        """格式化时间为 ASS 时间戳：H:MM:SS.CS（CS=1/100秒）"""
        try:
            total_cs = int(round(float(seconds) * 100.0))
        except Exception:
            total_cs = 0
        if total_cs < 0:
            total_cs = 0
        cs = total_cs % 100
        total_sec = total_cs // 100
        s = total_sec % 60
        m = (total_sec // 60) % 60
        h = total_sec // 3600
        return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

    @staticmethod
    def _ass_escape(text: str) -> str:
        """ASS 文本转义：换行->\\N，避免花括号被当作样式标签。"""
        text = '' if text is None else str(text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = text.replace('\n', r'\N')
        # 花括号可能触发 ASS override tags，直接替换为全角，避免破坏样式
        text = text.replace('{', '｛').replace('}', '｝')
        return text

    @staticmethod
    def _probe_video_resolution(video_path: Path) -> tuple[int | None, int | None]:
        """用 ffprobe 获取视频宽高；失败返回 (None, None)。"""
        try:
            out = subprocess.check_output(
                [
                    'ffprobe',
                    '-v', 'error',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=width,height',
                    '-of', 'json',
                    str(video_path),
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            data = json.loads(out) if out else {}
            streams = data.get('streams') or []
            if not streams:
                return None, None
            st = streams[0] or {}
            w = st.get('width')
            h = st.get('height')
            try:
                w = int(w) if w else None
            except Exception:
                w = None
            try:
                h = int(h) if h else None
            except Exception:
                h = None
            return w, h
        except Exception:
            return None, None

    def generate_srt(self, segments, output_path, translation_only=False):
        """生成SRT字幕文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, seg in enumerate(segments, 1):
                    start = self.format_time(seg['start'])
                    end = self.format_time(seg['end'])
                    text = seg['text']
                    translated = seg.get('translated', text)

                    f.write(f"{i}\n")
                    f.write(f"{start} --> {end}\n")

                    if translation_only:
                        f.write(f"{translated}\n\n")
                    else:
                        f.write(f"{text}\n")
                        f.write(f"{translated}\n\n")

            return True
        except Exception as e:
            logging.error(f"  × 生成字幕失败: {e}")
            return False

    def generate_ass(self, segments, output_path, translation_only=False, video_path: Path | None = None):
        """生成 ASS 字幕文件（V4+）。"""
        try:
            w, h = (None, None)
            if video_path is not None:
                w, h = self._probe_video_resolution(video_path)
            play_res_x = int(w or 1920)
            play_res_y = int(h or 1080)

            font_size = min(60, max(24, int(play_res_y * 0.06)))
            outline = max(2, int(font_size / 16))
            margin_v = max(20, int(font_size * 1.2))

            header = [
                "[Script Info]",
                "; Script generated by video-translation-service",
                "ScriptType: v4.00+",
                "Collisions: Normal",
                f"PlayResX: {play_res_x}",
                f"PlayResY: {play_res_y}",
                "WrapStyle: 2",
                "ScaledBorderAndShadow: yes",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                f"Style: Default,Noto Sans CJK SC,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,{outline},0,2,40,40,{margin_v},1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            ]

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(header) + "\n")
                for seg in segments:
                    start = self.format_time_ass(seg.get('start', 0.0))
                    end = self.format_time_ass(seg.get('end', 0.0))
                    text = self._ass_escape(seg.get('text', ''))
                    translated = self._ass_escape(seg.get('translated', seg.get('text', '')))
                    if translation_only:
                        line = translated
                    else:
                        line = f"{text}\\N{translated}"
                    f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{line}\n")
            return True
        except Exception as e:
            logging.error(f"  × 生成字幕失败: {e}")
            return False

    def generate_subtitle_file(self, segments, output_path: Path, translation_only: bool, video_path: Path):
        fmt = (self.subtitle_format or 'srt').strip().lower()
        if fmt == 'ass':
            return self.generate_ass(segments, str(output_path), translation_only, video_path=video_path)
        return self.generate_srt(segments, str(output_path), translation_only)

    def translate_video(self, video_path, target_lang='zh', source_lang='auto',
                        translation_only=False, output_dir=None):
        """翻译单个视频（带进度管理和并发润色）"""
        video_path = Path(video_path)
        video_name = video_path.name

        logging.info(f"\n{'=' * 70}")
        logging.info(f"处理: {video_path.name}")
        logging.info(f"{'=' * 70}")

        # 输出路径
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = video_path.parent

        fmt = (self.subtitle_format or 'srt').strip().lower()
        ext = '.ass' if fmt == 'ass' else '.srt'
        subtitle_path = output_dir / f"{video_path.stem}_{target_lang}{ext}"

        # 检查是否应该跳过
        should_skip, reason = self.should_skip_video(video_path, subtitle_path)
        if should_skip:
            logging.info(f"  跳过: {reason}")
            self.stats['skipped'] += 1
            return True
        elif reason:
            logging.info(f"  {reason}")

        # 标记为处理中
        self.update_video_status(video_name, 'processing')

        temp_files = []
        try:
            start_time = time.time()

            total_steps = 4
            if self.enable_vocal_separation:
                total_steps += 1
            if self.use_polish:
                total_steps += 1

            step = 1

            # 1. 提取音频
            logging.info(f"  [{step}/{total_steps}] 提取音频...")
            audio_path = output_dir / f"{video_path.stem}_temp.wav"
            audio_sep_path = output_dir / f"{video_path.stem}_temp_sep.wav"
            vocals_path = output_dir / f"{video_path.stem}_temp_vocals.wav"

            if self.enable_vocal_separation:
                # Demucs 建议使用较高采样率/双声道输入
                temp_files.append(audio_sep_path)
                if not self.extract_audio(str(video_path), str(audio_sep_path), sample_rate=44100, channels=2):
                    raise Exception("音频提取失败")
            else:
                temp_files.append(audio_path)
                if not self.extract_audio(str(video_path), str(audio_path)):
                    raise Exception("音频提取失败")
            logging.info(f"  [{step}/{total_steps}] 提取音频完成 ✓")

            # 2. 人声分离（可选）
            if self.enable_vocal_separation:
                step += 1
                self.clear_cuda_cache('before_vocal_separation')
                logging.info(f"  [{step}/{total_steps}] 人声分离（Demucs，可能较慢）...")
                temp_files.append(vocals_path)
                self.separate_vocals(audio_sep_path, vocals_path)
                asr_audio_path = vocals_path
                logging.info(f"  [{step}/{total_steps}] 人声分离完成 ✓")
            else:
                asr_audio_path = audio_path

            # 3. 语音识别
            step += 1
            self.clear_cuda_cache('before_asr')
            logging.info(f"  [{step}/{total_steps}] 语音识别（长视频可能需要数分钟）...")
            transcribe_start = time.time()
            asr_language = None if source_lang == 'auto' else source_lang
            if self.manage_models:
                # 避免与翻译模型同时占用显存
                if self.unload_models_after_tasks:
                    self._service_models_unload(['translation'])
                if not self.ensure_models(want_asr=True):
                    raise Exception("ASR模型加载失败")
            result = self.transcribe(str(asr_audio_path), language=asr_language)

            # 删除临时音频
            for temp_path in temp_files:
                try:
                    temp_path.unlink()
                except Exception:
                    pass

            if not result or not result.get('success'):
                raise Exception("语音识别失败")

            segments = result.get('segments', [])
            detected_lang = result.get('language', source_lang)
            transcribe_time = time.time() - transcribe_start
            logging.info(f"  [{step}/{total_steps}] 语音识别完成 ✓ ({len(segments)}段, {transcribe_time:.1f}秒)")
            if self.manage_models and self.unload_models_after_tasks:
                self._service_models_unload(['asr'])

            # 4. 翻译（批量翻译）
            step += 1
            translate_start = time.time()
            self.clear_cuda_cache('before_translation')
            logging.info(f"  [{step}/{total_steps}] 翻译 {len(segments)} 段...")
            if self.manage_models:
                # 确保 ASR 已释放（避免与翻译模型同时占用显存）
                if self.unload_models_after_tasks:
                    self._service_models_unload(['asr'])
                if not self.ensure_models(want_translation=True):
                    raise Exception("翻译模型加载失败")

            for i, seg in enumerate(segments, 1):
                translated = self.translate_text(
                    seg['text'],
                    detected_lang if source_lang == 'auto' else source_lang,
                    target_lang
                )
                seg['translated'] = translated

                # 每完成20%显示一次进度
                if i % max(1, len(segments) // 5) == 0 or i == len(segments):
                    logging.info(f"    翻译进度: {i}/{len(segments)} ({i * 100 // len(segments)}%)")

            logging.info(f"  [{step}/{total_steps}] 翻译完成 ✓")
            if self.manage_models and self.unload_models_after_tasks:
                self._service_models_unload(['translation'])

            # 5. 并发润色（可选）
            if self.use_polish:
                step += 1
                self.polish_batch_with_context(
                    segments,
                    detected_lang if source_lang == 'auto' else source_lang,
                    target_lang,
                    step_idx=step,
                    step_total=total_steps
                )

            translate_time = time.time() - translate_start
            polish_suffix = f" (含{self.concurrent_polish}线程并发润色)" if self.use_polish else ""

            # 6. 生成字幕
            step += 1
            logging.info(f"  [{step}/{total_steps}] 生成字幕...")
            if not self.generate_subtitle_file(segments, subtitle_path, translation_only, video_path):
                raise Exception("生成字幕失败")
            logging.info(f"  [{step}/{total_steps}] 生成字幕完成 ✓")

            total_time = time.time() - start_time

            logging.info(f"\n✓ 完成: {subtitle_path.name}")
            logging.info(f"  总耗时: {total_time:.1f}秒")
            logging.info(f"  语音识别: {transcribe_time:.1f}秒")
            logging.info(f"  翻译+润色: {translate_time:.1f}秒{polish_suffix}")

            # 标记为已完成
            self.update_video_status(
                video_name,
                'completed',
                srt_file=str(subtitle_path.name),  # 兼容旧字段名
                subtitle_file=str(subtitle_path.name),
                subtitle_format=fmt,
                duration=total_time
            )

            self.stats['success'] += 1
            return True

        except Exception as e:
            logging.error(f"\n× 处理失败: {e}")

            # 更新失败状态
            retry_count = self.progress_data.get(video_name, {}).get('retry_count', 0)
            self.update_video_status(
                video_name,
                'failed',
                error=str(e),
                retry_count=retry_count + 1
            )

            self.stats['failed'] += 1
            return False
        finally:
            # 失败时也尽量清理临时文件，避免超长视频遗留超大 WAV 占用磁盘
            for temp_path in temp_files:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def translate_directory(self, directory, target_lang='zh', source_lang='auto',
                            translation_only=False, recursive=False, output_dir=None):
        """批量翻译目录中的视频（带进度管理）"""
        directory = Path(directory)

        if not directory.exists():
            logging.error(f"× 目录不存在: {directory}")
            return

        # 查找视频文件
        if recursive:
            video_files = []
            for ext in self.video_extensions:
                video_files.extend(directory.rglob(f"*{ext}"))
        else:
            video_files = []
            for ext in self.video_extensions:
                video_files.extend(directory.glob(f"*{ext}"))

        video_files = sorted(video_files)

        if not video_files:
            logging.error(f"× 未找到视频文件（支持格式: {', '.join(self.video_extensions)}）")
            return

        # 生成任务名称
        task_name = directory.name.replace(' ', '_').replace('\\', '_').replace('/', '_')
        if not task_name:
            task_name = 'root'
        self.load_progress(task_name)

        self.stats['total'] = len(video_files)
        self.stats['start_time'] = time.time()

        logging.info(f"\n{'=' * 70}")
        logging.info(f"批量翻译任务: {task_name}")
        logging.info(f"{'=' * 70}")
        logging.info(f"目录: {directory}")
        logging.info(f"视频数量: {len(video_files)}")
        logging.info(f"目标语言: {target_lang}")
        logging.info(f"字幕模式: {'仅译文' if translation_only else '双语字幕'}")
        if self.use_polish:
            logging.info(f"DeepSeek润色: 启用（{self.concurrent_polish}线程并发）")
        else:
            logging.info(f"DeepSeek润色: 禁用")
        if self.enable_vocal_separation:
            logging.info(f"人声分离: 启用（Demucs {self.vocal_separation_model}, {self.vocal_separation_device}）")
        else:
            logging.info("人声分离: 禁用")
        logging.info(f"{'=' * 70}")

        # 处理每个视频
        for i, video_file in enumerate(video_files, 1):
            logging.info(f"\n[{i}/{len(video_files)}]")

            try:
                self.translate_video(
                    video_file,
                    target_lang,
                    source_lang,
                    translation_only,
                    output_dir
                )
            except KeyboardInterrupt:
                logging.info("\n\n用户中断 - 进度已保存，下次运行将继续")
                break
            except Exception as e:
                logging.error(f"× 意外错误: {e}")
                continue

        self.stats['end_time'] = time.time()
        self.print_report()

    def print_report(self):
        """打印处理报告"""
        if self.stats['start_time'] is None:
            return

        total_time = self.stats['end_time'] - self.stats['start_time']

        logging.info(f"\n{'=' * 70}")
        logging.info("处理报告")
        logging.info(f"{'=' * 70}")
        logging.info(f"总视频数: {self.stats['total']}")
        logging.info(f"成功: {self.stats['success']}")
        logging.info(f"失败: {self.stats['failed']}")
        logging.info(f"跳过: {self.stats['skipped']}")
        logging.info(f"总耗时: {total_time / 60:.1f}分钟")

        if self.stats['success'] > 0:
            avg_time = total_time / self.stats['success']
            logging.info(f"平均每个: {avg_time:.1f}秒")

        logging.info(f"{'=' * 70}")

    def show_progress(self, task_name):
        """显示进度"""
        self.load_progress(task_name)

        if not self.progress_data:
            logging.info("× 没有找到进度记录")
            return

        completed = [k for k, v in self.progress_data.items() if v.get('status') == 'completed']
        failed = [k for k, v in self.progress_data.items() if v.get('status') == 'failed']
        processing = [k for k, v in self.progress_data.items() if v.get('status') == 'processing']

        logging.info(f"\n{'=' * 70}")
        logging.info(f"进度报告: {task_name}")
        logging.info(f"{'=' * 70}")
        logging.info(f"已完成: {len(completed)}")
        logging.info(f"已失败: {len(failed)}")
        logging.info(f"处理中: {len(processing)}")
        logging.info(f"总计: {len(self.progress_data)}")
        logging.info(f"{'=' * 70}")

        if failed:
            logging.info("\n失败列表:")
            for video in failed[:10]:
                error = self.progress_data[video].get('error', '未知错误')
                retry = self.progress_data[video].get('retry_count', 0)
                logging.info(f"  - {video}: {error} (重试{retry}次)")
            if len(failed) > 10:
                logging.info(f"  ... 还有 {len(failed) - 10} 个失败")

    def reset_progress(self, task_name):
        """重置进度"""
        progress_file = self.progress_dir / f'{task_name}.json'
        if progress_file.exists():
            progress_file.unlink()
            logging.info(f"✓ 已清除进度: {task_name}")
        else:
            logging.info(f"× 没有找到进度文件: {task_name}")


def main():
    parser = argparse.ArgumentParser(
        description='批量视频翻译工具 v3.1 - 并发润色、上下文翻译、断点续传',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 翻译单个视频（10线程并发润色）
  python batch_translate.py video.mp4 -t zh

  # 批量翻译（自动断点续传）
  python batch_translate.py videos/ -t zh

  # 自定义并发数（20线程）
  python batch_translate.py videos/ -t zh --concurrent 20

  # 查看进度
  python batch_translate.py videos/ --show-progress
        """
    )

    parser.add_argument('input', help='视频文件或目录路径')
    parser.add_argument('-t', '--target', default='zh', help='目标语言（默认: zh）')
    parser.add_argument('-s', '--source', default=None, help='源语言/ASR语言（默认: 读取config.ini；未配置则auto自动检测）')
    parser.add_argument('-o', '--output', help='输出目录（默认: 与视频同目录）')
    parser.add_argument('--translation-only', action='store_true', help='仅生成译文字幕（不含原文）')
    parser.add_argument('--recursive', '-r', action='store_true', help='递归处理子目录')
    parser.add_argument('--polish', action='store_true', help='使用DeepSeek润色翻译')
    parser.add_argument('--concurrent', type=int, default=10, help='DeepSeek并发数（默认: 10）')
    parser.add_argument('--deepseek-key', help='DeepSeek API密钥')
    parser.add_argument('--service-url', default='http://127.0.0.1:50515',
                        help='翻译服务地址（默认: http://127.0.0.1:50515）')
    parser.add_argument('--vocal-separation', action='store_true', default=None,
                        help='启用人声分离（Demucs，用于背景音乐/嘈杂场景；需要 pip install demucs）')
    parser.add_argument('--vocal-model', default=None,
                        help='Demucs 模型名（默认: 读取config.ini 或 htdemucs）')
    parser.add_argument('--vocal-device', default=None, choices=['auto', 'cpu', 'cuda'],
                        help='人声分离设备：auto/cpu/cuda（默认: 读取config.ini 或 auto）')
    parser.add_argument('--vocal-chunk-sec', type=int, default=None,
                        help='Demucs 分段秒数（仅超长/超大音频触发；默认读取config.ini；默认1800）')
    parser.add_argument('--cuda-clear', dest='cuda_clear', action='store_true', default=None,
                        help='在GPU重任务前清理CUDA缓存（降低OOM概率，略慢）')
    parser.add_argument('--no-cuda-clear', dest='cuda_clear', action='store_false',
                        help='禁用GPU任务前清理CUDA缓存')
    parser.add_argument('--asr-chunk-sec', type=int, default=None,
                        help='ASR 分块秒数（启用进度条/降低长音频500/OOM；0=禁用；默认读取config.ini）')
    parser.add_argument('--asr-overlap-sec', type=float, default=None,
                        help='ASR 分块重叠秒数（避免切在单词中间；默认读取config.ini）')
    parser.add_argument('--manage-models', dest='manage_models', action='store_true', default=None,
                        help='按需加载/卸载服务端模型（适合显存紧张环境，如 Colab T4）')
    parser.add_argument('--no-manage-models', dest='manage_models', action='store_false',
                        help='禁用按需加载/卸载服务端模型')
    parser.add_argument('--unload-models', dest='unload_models', action='store_true', default=None,
                        help='在 ASR/翻译完成后卸载服务端模型释放显存（需要 --manage-models 或 config.ini）')
    parser.add_argument('--no-unload-models', dest='unload_models', action='store_false',
                        help='不在任务后卸载模型')
    parser.add_argument('--model-load-timeout', type=int, default=None,
                        help='等待服务端模型加载超时秒数（默认读取config.ini或3600）')
    parser.add_argument('--wait-ready', action='store_true',
                        help='等待翻译服务就绪（首次下载/加载模型可能需要较长时间）')
    parser.add_argument('--wait-timeout', type=int, default=3600,
                        help='等待翻译服务就绪超时秒数（默认: 3600）')
    parser.add_argument('--show-progress', action='store_true', help='显示当前进度')
    parser.add_argument('--reset-progress', action='store_true', help='清除进度记录，从头开始')
    parser.add_argument('--subtitle-format', choices=['srt', 'ass'], default=None,
                        help='字幕格式：srt/ass（默认读取config.ini；未配置则srt）')

    args = parser.parse_args()

    # 检查输入
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"× 路径不存在: {args.input}")
        return 1

    # 生成任务名称
    if input_path.is_dir():
        task_name = input_path.name.replace(' ', '_').replace('\\', '_').replace('/', '_')
        if not task_name:
            task_name = 'root'
    else:
        task_name = input_path.parent.name.replace(' ', '_').replace('\\', '_').replace('/', '_')
        if not task_name:
            task_name = 'root'

    # 确定是否使用润色
    use_polish = args.polish
    if not use_polish and CONFIG_AVAILABLE:
        use_polish = config.use_deepseek_polish

    # 源语言（命令行 > config.ini > auto）
    if args.source:
        source_lang = args.source
    elif CONFIG_AVAILABLE:
        source_lang = getattr(config, 'asr_language', None) or 'auto'
    else:
        source_lang = 'auto'

    # 确定是否启用人声分离（命令行 > config.ini）
    if args.vocal_separation is None:
        enable_vocal_separation = config.enable_vocal_separation if CONFIG_AVAILABLE else False
    else:
        enable_vocal_separation = bool(args.vocal_separation)

    vocal_separation_model = args.vocal_model
    if not vocal_separation_model and CONFIG_AVAILABLE:
        vocal_separation_model = config.vocal_separation_model
    if not vocal_separation_model:
        vocal_separation_model = 'htdemucs'

    vocal_separation_device = args.vocal_device
    if not vocal_separation_device and CONFIG_AVAILABLE:
        vocal_separation_device = config.vocal_separation_device
    if not vocal_separation_device:
        vocal_separation_device = 'auto'

    # Demucs 分段秒数（命令行 > config.ini > 1800）
    if args.vocal_chunk_sec is None:
        vocal_separation_chunk_sec = config.vocal_separation_chunk_sec if CONFIG_AVAILABLE else 1800
    else:
        vocal_separation_chunk_sec = int(args.vocal_chunk_sec)
    if vocal_separation_chunk_sec <= 0:
        vocal_separation_chunk_sec = 1800

    # 是否在 GPU 重任务前清理 CUDA 缓存（命令行 > config.ini）
    if args.cuda_clear is None:
        clear_cuda_cache_before_tasks = config.clear_cuda_cache_before_tasks if CONFIG_AVAILABLE else False
    else:
        clear_cuda_cache_before_tasks = bool(args.cuda_clear)

    # ASR 分块（命令行 > config.ini）
    if args.asr_chunk_sec is None:
        asr_chunk_sec = config.asr_chunk_sec if CONFIG_AVAILABLE else 0
    else:
        asr_chunk_sec = int(args.asr_chunk_sec)

    if args.asr_overlap_sec is None:
        asr_chunk_overlap_sec = config.asr_chunk_overlap_sec if CONFIG_AVAILABLE else 0.0
    else:
        asr_chunk_overlap_sec = float(args.asr_overlap_sec)

    # 按需加载/卸载服务端模型（命令行 > config.ini）
    if args.manage_models is None:
        manage_models = config.manage_models if CONFIG_AVAILABLE else False
    else:
        manage_models = bool(args.manage_models)

    if args.unload_models is None:
        unload_models_after_tasks = config.unload_models_after_tasks if CONFIG_AVAILABLE else False
    else:
        unload_models_after_tasks = bool(args.unload_models)

    if unload_models_after_tasks:
        manage_models = True

    if args.model_load_timeout is None:
        model_load_timeout = 3600
    else:
        model_load_timeout = int(args.model_load_timeout)

    # 字幕格式（命令行 > config.ini > srt）
    if args.subtitle_format:
        subtitle_format = (args.subtitle_format or 'srt').strip().lower()
    elif CONFIG_AVAILABLE:
        subtitle_format = getattr(config, 'subtitle_format', 'srt') or 'srt'
    else:
        subtitle_format = 'srt'

    # 创建翻译器
    translator = VideoTranslator(
        service_url=args.service_url,
        deepseek_key=args.deepseek_key,
        use_polish=use_polish,
        concurrent_polish=args.concurrent,
        enable_vocal_separation=enable_vocal_separation,
        vocal_separation_model=vocal_separation_model,
        vocal_separation_device=vocal_separation_device,
        vocal_separation_chunk_sec=vocal_separation_chunk_sec,
        clear_cuda_cache_before_tasks=clear_cuda_cache_before_tasks,
        asr_chunk_sec=asr_chunk_sec,
        asr_chunk_overlap_sec=asr_chunk_overlap_sec,
        manage_models=manage_models,
        unload_models_after_tasks=unload_models_after_tasks,
        model_load_timeout=model_load_timeout,
        subtitle_format=subtitle_format,
    )

    # 处理进度命令
    if args.show_progress:
        translator.show_progress(task_name)
        return 0

    if args.reset_progress:
        translator.reset_progress(task_name)
        if not input_path.exists():
            return 0

    # 设置日志系统
    log_file = setup_logger()
    logging.info(f"日志文件: {log_file}")
    logging.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("")

    # 自动清理旧日志（如果配置启用）
    if CONFIG_AVAILABLE:
        auto_cleanup = getattr(config, 'auto_cleanup_logs', False)
        keep_days = getattr(config, 'log_keep_days', 7)
    else:
        auto_cleanup = os.getenv('AUTO_CLEANUP_LOGS', '').lower() in ('true', '1', 'yes')
        keep_days = int(os.getenv('LOG_KEEP_DAYS', '7'))

    if auto_cleanup:
        cleanup_old_logs('log', keep_days, auto_cleanup)

    # 检查服务
    if not translator.check_service(wait_ready=args.wait_ready, wait_timeout=args.wait_timeout):
        return 1

    # 检查人声分离依赖
    if enable_vocal_separation:
        try:
            import demucs  # noqa: F401
        except Exception:
            logging.error("× 已启用人声分离，但未安装 demucs")
            logging.error("  请运行: pip install demucs")
            return 1

    # 显示配置信息
    if use_polish or args.polish:
        if translator.deepseek_key:
            logging.info(f"✓ DeepSeek API密钥已配置")
            if translator.use_polish:
                logging.info(f"✓ DeepSeek并发润色已启用（{args.concurrent}线程）")
        else:
            logging.error("× DeepSeek API密钥未配置")

    # 检查DeepSeek配置
    if (args.polish or use_polish) and not translator.deepseek_key:
        logging.error("× 启用润色功能需要DeepSeek API密钥")
        logging.error("  方法1: 在 config.ini 中配置 [API] deepseek_api_key")
        logging.error("  方法2: 设置环境变量 set DEEPSEEK_API_KEY=your_key")
        logging.error("  方法3: 使用参数 --deepseek-key your_key")
        return 1

    logging.info("")

    # 开始处理
    try:
        if input_path.is_file():
            # 单个视频
            translator.load_progress(task_name)
            translator.stats['total'] = 1
            translator.stats['start_time'] = time.time()

            translator.translate_video(
                input_path,
                args.target,
                source_lang,
                args.translation_only,
                args.output
            )

            translator.stats['end_time'] = time.time()
            translator.print_report()
        else:
            # 目录批量处理
            translator.translate_directory(
                input_path,
                args.target,
                source_lang,
                args.translation_only,
                args.recursive,
                args.output
            )
    except KeyboardInterrupt:
        logging.info("\n\n用户中断")
        return 1

    logging.info(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"日志已保存到: {log_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
