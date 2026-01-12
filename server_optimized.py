import os
import io
import time
import sys
import threading
import traceback
from contextlib import nullcontext as _nullcontext
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
from typing import Optional

# 强制输出无缓冲
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)

# Lazy imports to speed cold start
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, M2M100ForConditionalGeneration, M2M100Tokenizer
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import torch
except ImportError:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    M2M100ForConditionalGeneration = None
    M2M100Tokenizer = None
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None
    pipeline = None
    torch = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_LOCK = threading.Lock()
ASR_MODEL = None
ASR_BACKEND = 'faster-whisper'
ASR_MODEL_NAME = None
TRANSLATION_MODEL = None
TOKENIZER = None

# Defaults are populated in __main__ (and used by /models/load when payload omits fields)
SERVICE_CONFIG = {
    'asr_model_size': 'reazonspeech',
    'translation_model': 'facebook/nllb-200-1.3B',
    'use_gpu': True,
    'device': 'cuda' if (torch and hasattr(torch, 'cuda') and torch.cuda.is_available()) else 'cpu',
    'compute_type': 'float16' if (torch and hasattr(torch, 'cuda') and torch.cuda.is_available()) else 'int8',
    'asr_transformers_chunk_sec': 30,
    'asr_transformers_stride_sec': 5.0,
}

DEFAULT_REAZON_ASR_MODEL = 'japanese-asr/distil-whisper-large-v3-ja-reazonspeech-large'


def _resolve_asr_backend(asr_size: str) -> tuple[str, str]:
    """返回 (backend, model_id_or_size)；reazonspeech 走 transformers，其它走 faster-whisper。"""
    raw = (asr_size or '').strip()
    if not raw:
        return 'transformers', DEFAULT_REAZON_ASR_MODEL
    low = raw.lower()
    if low.startswith('reazonspeech:'):
        model_id = raw.split(':', 1)[1].strip() or DEFAULT_REAZON_ASR_MODEL
        return 'transformers', model_id
    if low in ('reazonspeech', 'reazon', 'rs'):
        return 'transformers', DEFAULT_REAZON_ASR_MODEL
    return 'faster-whisper', raw


def _get_transformers_chunk_settings() -> tuple[float, float] | None:
    """获取 transformers ASR 内部分块配置（<=0 表示禁用）。"""
    try:
        chunk_sec = float(SERVICE_CONFIG.get('asr_transformers_chunk_sec', 30) or 0)
    except Exception:
        chunk_sec = 30.0
    try:
        stride_sec = float(SERVICE_CONFIG.get('asr_transformers_stride_sec', 5.0) or 0.0)
    except Exception:
        stride_sec = 5.0
    if chunk_sec <= 0:
        return None
    if stride_sec < 0:
        stride_sec = 0.0
    if stride_sec >= chunk_sec:
        stride_sec = max(0.0, chunk_sec - 0.1)
    return chunk_sec, stride_sec


def _want_traceback() -> bool:
    v = os.environ.get('VTS_RETURN_TRACEBACK') or os.environ.get('RETURN_TRACEBACK') or ''
    return v.strip().lower() in ('1', 'true', 'yes', 'on')


@app.errorhandler(Exception)
def handle_unhandled_exception(e):
    """保证返回 JSON，避免客户端只看到 HTML 500 页面。"""
    if isinstance(e, HTTPException):
        payload = {
            'success': False,
            'error': e.description,
            'type': type(e).__name__,
            'code': e.code,
        }
        if _want_traceback():
            payload['traceback'] = traceback.format_exc()[-8000:]
        return jsonify(payload), e.code

    payload = {
        'success': False,
        'error': str(e),
        'type': type(e).__name__,
        'code': 500,
    }
    if _want_traceback():
        payload['traceback'] = traceback.format_exc()[-8000:]
    return jsonify(payload), 500


def _torch_cuda_available() -> bool:
    try:
        return bool(torch and torch.cuda.is_available())
    except Exception:
        return False

# Readiness booleans kept for backward compatibility
ASR_READY = False
TRANSLATION_READY = False

# Extended status object for progress reporting
STATUS = {
    'phase': 'idle',               # idle | asr_downloading | asr_loading | asr_ready | translation_downloading | translation_loading | translation_ready | ready | error
    'progress': 0.0,               # 0.0 ~ 1.0 coarse grained
    'message': 'Idle',
    'started_at': None,
    'error': None,
}

LOADING_THREAD: Optional[threading.Thread] = None
LOADING_PARAMS = {}

def _set_status(phase: str, progress: float, message: str):
    STATUS['phase'] = phase
    STATUS['progress'] = round(max(0.0, min(1.0, progress)), 3)
    STATUS['message'] = message
    STATUS['error'] = None if phase != 'error' else STATUS.get('error')

def _set_error(err_msg: str):
    STATUS['phase'] = 'error'
    STATUS['message'] = err_msg
    STATUS['error'] = err_msg

def _load_models_async(asr_size: str, translation_name: str, device: str, compute_type: str,
                       load_asr: bool = True, load_translation: bool = True):
    """Background thread target to load models with phase updates."""
    global ASR_MODEL, ASR_BACKEND, ASR_MODEL_NAME, TRANSLATION_MODEL, TOKENIZER, ASR_READY, TRANSLATION_READY, LOADING_THREAD
    try:
        if STATUS['started_at'] is None:
            STATUS['started_at'] = time.time()
        # ASR model
        if load_asr and not ASR_READY:
            asr_backend, asr_id = _resolve_asr_backend(asr_size)
            ASR_BACKEND = asr_backend
            ASR_MODEL_NAME = asr_id
            _set_status('asr_downloading', 0.02, f'Downloading/Preparing ASR model {asr_id} ...')
            try:
                if asr_backend == 'transformers':
                    if not (AutoModelForSpeechSeq2Seq and AutoProcessor and pipeline and torch):
                        raise ImportError('Transformers ASR not available')
                    _set_status('asr_loading', 0.08, f'Loading ASR model {asr_id} (transformers) ...')
                    dtype = torch.float16 if device == 'cuda' else torch.float32
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        asr_id,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        cache_dir='./models/reazonspeech'
                    )
                    if device == 'cuda':
                        model = model.to(device)
                    processor = AutoProcessor.from_pretrained(asr_id, cache_dir='./models/reazonspeech')
                    ASR_MODEL = pipeline(
                        'automatic-speech-recognition',
                        model=model,
                        tokenizer=processor.tokenizer,
                        feature_extractor=processor.feature_extractor,
                        device=0 if device == 'cuda' else -1
                    )
                else:
                    if not WhisperModel:
                        raise ImportError('faster-whisper not available')
                    # faster-whisper downloads inside constructor
                    _set_status('asr_loading', 0.08, f'Loading ASR model {asr_id} (faster-whisper) ...')
                    ASR_MODEL = WhisperModel(asr_id, device=device, compute_type=compute_type, download_root='./models/whisper')
                ASR_READY = True
                _set_status('asr_ready', 0.35, f'ASR model ready ({asr_id}).')
            except Exception as e:
                _set_error(f'ASR load failed: {e}')
                return
        elif load_asr and ASR_READY:
            _set_status('asr_ready', max(STATUS['progress'], 0.35), 'ASR already ready.')
        # Translation model - 使用m2m100或NLLB
        if load_translation and not TRANSLATION_READY:
            # 检测是否是m2m100模型
            is_m2m100 = 'm2m100' in translation_name.lower()
            _set_status('translation_downloading', 0.4, f'Downloading translation model {translation_name} ...')
            try:
                if is_m2m100 and M2M100Tokenizer and M2M100ForConditionalGeneration:
                    print(f'[Translation] Using M2M100 model: {translation_name}')
                    TOKENIZER = M2M100Tokenizer.from_pretrained(translation_name, cache_dir='./models/m2m100')
                    _set_status('translation_loading', 0.55, f'Loading translation model {translation_name} ...')
                    TRANSLATION_MODEL = M2M100ForConditionalGeneration.from_pretrained(translation_name, cache_dir='./models/m2m100')
                elif AutoTokenizer and AutoModelForSeq2SeqLM:
                    print(f'[Translation] Using NLLB model: {translation_name}')
                    TOKENIZER = AutoTokenizer.from_pretrained(translation_name, cache_dir='./models/nllb')
                    _set_status('translation_loading', 0.55, f'Loading translation model {translation_name} ...')
                    TRANSLATION_MODEL = AutoModelForSeq2SeqLM.from_pretrained(translation_name, cache_dir='./models/nllb')
                else:
                    raise ImportError('No translation library available')
                    
                if torch and device == 'cuda':
                    TRANSLATION_MODEL = TRANSLATION_MODEL.to(device)
                    TRANSLATION_MODEL.eval()
                TRANSLATION_READY = True
                _set_status('translation_ready', 0.85, f'Translation model ready ({translation_name}).')
                print(f'[Translation] Model loaded successfully: {translation_name}')
            except Exception as e:
                _set_error(f'Translation load failed: {e}')
                print(f'[Translation] Error loading model: {e}')
                import traceback
                traceback.print_exc()
                return
        elif load_translation and TRANSLATION_READY:
            _set_status('translation_ready', max(STATUS['progress'], 0.85), 'Translation already ready.')
        if ASR_READY and TRANSLATION_READY:
            _set_status('ready', 1.0, 'All models ready.')
    finally:
        LOADING_THREAD = None

# Simple health endpoint
@app.route('/health', methods=['GET'])
def health():
    # Backward compatible booleans plus extended status
    return jsonify({
        'status': 'ok',
        'asr_ready': ASR_READY,
        'translation_ready': TRANSLATION_READY,
        'asr_backend': ASR_BACKEND,
        'asr_model': ASR_MODEL_NAME,
        'phase': STATUS['phase'],
        'progress': STATUS['progress'],
        'message': STATUS['message'],
        'error': STATUS['error'],
        'ready': ASR_READY and TRANSLATION_READY
    })

@app.route('/gpu/clear', methods=['POST'])
def gpu_clear():
    """清理 CUDA 缓存（用于降低 OOM 概率）"""
    if not torch or not torch.cuda.is_available():
        return jsonify({'success': True, 'cuda_available': False})

    payload = request.json or {}
    stage = payload.get('stage') or payload.get('label') or payload.get('reason')

    with MODEL_LOCK:
        try:
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return jsonify({
                'success': True,
                'cuda_available': True,
                'stage': stage,
                'free_bytes': int(free_bytes),
                'total_bytes': int(total_bytes),
                'allocated_bytes': int(torch.cuda.memory_allocated()),
                'reserved_bytes': int(torch.cuda.memory_reserved()),
            })
        except Exception as e:
            return jsonify({'success': False, 'cuda_available': True, 'stage': stage, 'error': str(e)}), 500

@app.route('/init', methods=['POST'])
def init_models():
    global LOADING_THREAD, LOADING_PARAMS
    payload = request.json or {}
    asr_size = payload.get('asr_model_size') or SERVICE_CONFIG.get('asr_model_size', 'reazonspeech')
    translation_name = payload.get('translation_model') or SERVICE_CONFIG.get('translation_model', 'facebook/nllb-200-1.3B')
    want_gpu = payload.get('use_gpu', True)
    device = 'cuda' if want_gpu and _torch_cuda_available() else 'cpu'
    compute_type = payload.get('compute_type', 'float16' if device == 'cuda' else 'int8')

    # If already ready return immediately
    if ASR_READY and TRANSLATION_READY:
        return jsonify({'success': True, 'already_ready': True, 'asr_ready': ASR_READY, 'translation_ready': TRANSLATION_READY, 'phase': STATUS['phase'], 'progress': STATUS['progress']})

    # Avoid spawning multiple threads
    with MODEL_LOCK:
        if LOADING_THREAD is None:
            _set_status('starting', 0.0, 'Starting model load thread ...')
            LOADING_PARAMS = {
                'asr_size': asr_size,
                'translation_name': translation_name,
                'device': device,
                'compute_type': compute_type,
            }
            LOADING_THREAD = threading.Thread(target=_load_models_async, args=(asr_size, translation_name, device, compute_type), daemon=True)
            LOADING_THREAD.start()
        else:
            # Thread already running; optionally update desired params if changed
            LOADING_PARAMS.update({'asr_size': asr_size, 'translation_name': translation_name})
    return jsonify({'success': True, 'started': True, 'phase': STATUS['phase'], 'progress': STATUS['progress']})


@app.route('/models/load', methods=['POST'])
def models_load():
    """
    按需加载模型（避免启动时把 ASR + 翻译同时塞进显存）。

    payload:
      {
        "models": ["asr"] | ["translation"] | ["asr","translation"],
        "asr_model_size": "...",           # optional
        "translation_model": "...",        # optional
        "use_gpu": true/false,             # optional
        "compute_type": "float16|int8"     # optional (ASR)
      }
    """
    global LOADING_THREAD, LOADING_PARAMS
    payload = request.json or {}
    models = payload.get('models') or payload.get('model') or []
    if isinstance(models, str):
        models = [models]
    if not isinstance(models, list):
        return jsonify({'success': False, 'error': 'Invalid models'}), 400

    want_asr = any(str(m).lower() in ('asr', 'whisper') for m in models)
    want_translation = any(str(m).lower() in ('translation', 'translate', 'nllb', 'm2m100') for m in models)
    if not want_asr and not want_translation:
        return jsonify({'success': False, 'error': 'No models requested'}), 400

    asr_size = payload.get('asr_model_size') or SERVICE_CONFIG.get('asr_model_size', 'reazonspeech')
    translation_name = payload.get('translation_model') or payload.get('translation_name') or SERVICE_CONFIG.get('translation_model', 'facebook/nllb-200-1.3B')
    want_gpu = payload.get('use_gpu')
    if want_gpu is None:
        want_gpu = SERVICE_CONFIG.get('use_gpu', True)
    device = 'cuda' if bool(want_gpu) and _torch_cuda_available() else 'cpu'
    compute_type = payload.get('compute_type') or ('float16' if device == 'cuda' else 'int8')

    # If already ready for requested models, return immediately
    if (not want_asr or ASR_READY) and (not want_translation or TRANSLATION_READY):
        return jsonify({'success': True, 'already_ready': True, 'asr_ready': ASR_READY, 'translation_ready': TRANSLATION_READY, 'phase': STATUS['phase'], 'progress': STATUS['progress']})

    with MODEL_LOCK:
        if LOADING_THREAD is None:
            _set_status('starting', 0.0, f"Starting model load thread ({'asr' if want_asr else ''}{'+' if want_asr and want_translation else ''}{'translation' if want_translation else ''}) ...")
            LOADING_PARAMS = {
                'asr_size': asr_size,
                'translation_name': translation_name,
                'device': device,
                'compute_type': compute_type,
                'load_asr': want_asr,
                'load_translation': want_translation,
            }
            LOADING_THREAD = threading.Thread(
                target=_load_models_async,
                args=(asr_size, translation_name, device, compute_type, want_asr, want_translation),
                daemon=True
            )
            LOADING_THREAD.start()
        else:
            # Thread already running; do not spawn another one.
            pass

    return jsonify({'success': True, 'started': True, 'asr_ready': ASR_READY, 'translation_ready': TRANSLATION_READY, 'phase': STATUS['phase'], 'progress': STATUS['progress']})


@app.route('/models/unload', methods=['POST'])
def models_unload():
    """
    卸载模型以释放显存。

    payload:
      { "models": ["asr"] | ["translation"] | ["all"] }
    """
    global ASR_MODEL, ASR_BACKEND, ASR_MODEL_NAME, TRANSLATION_MODEL, TOKENIZER, ASR_READY, TRANSLATION_READY
    payload = request.json or {}
    models = payload.get('models') or payload.get('model') or ['all']
    if isinstance(models, str):
        models = [models]
    if not isinstance(models, list):
        return jsonify({'success': False, 'error': 'Invalid models'}), 400

    want_all = any(str(m).lower() in ('all', '*') for m in models)
    unload_asr = want_all or any(str(m).lower() in ('asr', 'whisper') for m in models)
    unload_translation = want_all or any(str(m).lower() in ('translation', 'translate', 'nllb', 'm2m100') for m in models)

    with MODEL_LOCK:
        if unload_asr:
            ASR_MODEL = None
            ASR_BACKEND = 'faster-whisper'
            ASR_MODEL_NAME = None
            ASR_READY = False
        if unload_translation:
            TRANSLATION_MODEL = None
            TOKENIZER = None
            TRANSLATION_READY = False

        # Best-effort CUDA memory release
        try:
            import gc

            gc.collect()
        except Exception:
            pass
        try:
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

        if ASR_READY and TRANSLATION_READY:
            _set_status('ready', 1.0, 'All models ready.')
        elif ASR_READY:
            _set_status('asr_ready', 0.35, 'ASR model ready.')
        elif TRANSLATION_READY:
            _set_status('translation_ready', 0.85, 'Translation model ready.')
        else:
            _set_status('idle', 0.0, 'No models loaded.')

        free_bytes = None
        total_bytes = None
        allocated_bytes = None
        reserved_bytes = None
        try:
            if torch and torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                allocated_bytes = int(torch.cuda.memory_allocated())
                reserved_bytes = int(torch.cuda.memory_reserved())
        except Exception:
            pass

    return jsonify({
        'success': True,
        'asr_ready': ASR_READY,
        'translation_ready': TRANSLATION_READY,
        'cuda_available': bool(torch and torch.cuda.is_available()) if torch else False,
        'free_bytes': int(free_bytes) if free_bytes is not None else None,
        'total_bytes': int(total_bytes) if total_bytes is not None else None,
        'allocated_bytes': allocated_bytes,
        'reserved_bytes': reserved_bytes,
    })

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if not ASR_READY:
        return jsonify({'success': False, 'error': 'ASR not ready'}), 400
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file'}), 400
    f = request.files['audio']
    # ✅ 生成唯一文件名避免并发冲突
    import uuid
    ext = os.path.splitext(secure_filename(f.filename or 'audio.bin'))[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    language = request.form.get('language')  # optional
    started = time.time()
    try:
        f.save(path)
        # 如果是PCM，转换为WAV
        if filename.endswith('.pcm') or filename.endswith('.raw'):
            import wave
            wav_path = path.replace('.pcm', '.wav').replace('.raw', '.wav')
            with wave.open(wav_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                with open(path, 'rb') as pcm_file:
                    wav_file.writeframes(pcm_file.read())
            os.remove(path)
            path = wav_path
        
        # ✅ 使用锁防止与 /models/unload 并发导致崩溃
        with MODEL_LOCK:
            if ASR_BACKEND == 'transformers':
                gen_kwargs = {'task': 'transcribe'}
                if language and language != 'auto':
                    gen_kwargs['language'] = language
                pipe_kwargs = {
                    'return_timestamps': True,
                    'generate_kwargs': gen_kwargs,
                }
                chunk_settings = _get_transformers_chunk_settings()
                if chunk_settings:
                    chunk_sec, stride_sec = chunk_settings
                    pipe_kwargs['chunk_length_s'] = chunk_sec
                    if stride_sec:
                        pipe_kwargs['stride_length_s'] = stride_sec
                    pipe_kwargs['batch_size'] = 1
                result = ASR_MODEL(path, **pipe_kwargs)
                text = (result.get('text') or '').strip()
                chunks = result.get('chunks') or result.get('segments') or []
                seg_list = []
                for ch in chunks:
                    ts = ch.get('timestamp') or ch.get('timestamps')
                    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                        start, end = ts[0], ts[1]
                    else:
                        start = ch.get('start')
                        end = ch.get('end')
                    if start is None:
                        start = 0.0
                    if end is None:
                        end = start
                    seg_list.append({
                        'start': float(start),
                        'end': float(end),
                        'text': (ch.get('text') or '').strip()
                    })
                info_language = language or 'unknown'
                info_prob = None
            else:
                segments, info = ASR_MODEL.transcribe(
                    path,
                    language=language,
                    vad_filter=True,  # 使用VAD检测静音点，在语音边界截断
                    beam_size=5,
                    best_of=5
                )
                text = ''
                seg_list = []
                for s in segments:
                    text += s.text
                    seg_list.append({'start': s.start, 'end': s.end, 'text': s.text})
                info_language = info.language
                info_prob = info.language_probability
        os.remove(path)
        return jsonify({'success': True, 'text': text.strip(), 'language': info_language, 'language_probability': info_prob, 'segments': seg_list, 'processing_time_ms': int((time.time()-started)*1000)})
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate():
    if not TRANSLATION_READY:
        return jsonify({'success': False, 'error': 'Translation not ready'}), 400
    data = request.json or {}
    text = data.get('text')
    src_lang = data.get('source_language')
    tgt_lang = data.get('target_language')
    if not text:
        return jsonify({'success': False, 'error': 'No text'}), 400
    if not src_lang or not tgt_lang:
        return jsonify({'success': False, 'error': 'Missing languages'}), 400
    
    # 语言代码映射 - 支持NLLB和M2M100
    # M2M100使用简单的ISO代码（zh, en等）
    # NLLB使用扩展代码（zho_Hans, eng_Latn等）
    nllb_lang_map = {
        'zh': 'zho_Hans', 'en': 'eng_Latn', 'ja': 'jpn_Jpan', 
        'ko': 'kor_Hang', 'de': 'deu_Latn', 'fr': 'fra_Latn',
        'es': 'spa_Latn', 'ru': 'rus_Cyrl', 'ar': 'arb_Arab'
    }
    m2m100_lang_map = {
        'zh': 'zh', 'en': 'en', 'ja': 'ja', 'ko': 'ko', 
        'de': 'de', 'fr': 'fr', 'es': 'es', 'ru': 'ru', 'ar': 'ar'
    }
    
    # 检测模型类型
    is_m2m100 = isinstance(TOKENIZER, M2M100Tokenizer) if M2M100Tokenizer else False
    
    try:
        # 使用锁防止并发访问模型（修复"Already borrowed"错误）
        with MODEL_LOCK:
            with (torch.inference_mode() if torch else _nullcontext()):
                if is_m2m100:
                    # M2M100翻译流程
                    src_code = m2m100_lang_map.get(src_lang, src_lang)
                    tgt_code = m2m100_lang_map.get(tgt_lang, tgt_lang)
                    TOKENIZER.src_lang = src_code
                    inputs = TOKENIZER(text, return_tensors='pt')
                    if torch and TRANSLATION_MODEL and torch.cuda.is_available() and next(TRANSLATION_MODEL.parameters()).is_cuda:
                        inputs = {k: v.to('cuda') for k, v in inputs.items()}
                    # M2M100使用forced_bos_token_id指定目标语言
                    tgt_lang_id = TOKENIZER.get_lang_id(tgt_code)
                    gen_tokens = TRANSLATION_MODEL.generate(
                        **inputs,
                        forced_bos_token_id=tgt_lang_id,
                        max_length=512,
                        num_beams=3,
                        early_stopping=True
                    )
                    decoded = TOKENIZER.batch_decode(gen_tokens, skip_special_tokens=True)
                else:
                    # NLLB翻译流程
                    src_code = nllb_lang_map.get(src_lang, src_lang)
                    tgt_code = nllb_lang_map.get(tgt_lang, tgt_lang)
                    TOKENIZER.src_lang = src_code
                    inputs = TOKENIZER(text, return_tensors='pt')
                    if torch and TRANSLATION_MODEL and torch.cuda.is_available() and next(TRANSLATION_MODEL.parameters()).is_cuda:
                        inputs = {k: v.to('cuda') for k, v in inputs.items()}
                    gen_tokens = TRANSLATION_MODEL.generate(
                        **inputs,
                        forced_bos_token_id=TOKENIZER.convert_tokens_to_ids(tgt_code),
                        max_length=512,
                        num_beams=3
                    )
                    decoded = TOKENIZER.batch_decode(gen_tokens, skip_special_tokens=True)
        
        if not decoded:
            return jsonify({'success': False, 'error': 'Translation produced empty result'}), 500
        translated = decoded[0]
        return jsonify({'success': True, 'translated_text': translated})
    except Exception as e:
        import traceback
        error_msg = f"Translation error: {str(e)}"
        print(f"[Translation Error] {error_msg}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': error_msg}), 500

@app.route('/process', methods=['POST'])
def process():
    # full pipeline: transcribe + translate
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file'}), 400
    tgt_lang = request.form.get('target_language', 'en')
    src_lang_override = request.form.get('source_language')
    # ✅ 如果source_language是'auto'，不使用override
    if src_lang_override == 'auto':
        src_lang_override = None
    # Step ASR
    trans_resp = transcribe()
    if isinstance(trans_resp, tuple):
        return trans_resp  # error
    data = trans_resp.get_json()
    if not data.get('success'):
        return trans_resp
    source_lang = src_lang_override or data.get('language')
    if source_lang == tgt_lang:
        return jsonify({'success': True, 'source_text': data.get('text'), 'translated_text': data.get('text'), 'source_language': source_lang, 'target_language': tgt_lang, 'segments': data.get('segments')})
    # Translation
    if TRANSLATION_READY:
        # 构造翻译请求数据
        translate_data = {
            'text': data.get('text'),
            'source_language': source_lang,
            'target_language': tgt_lang
        }
        # 调用翻译逻辑（支持M2M100和NLLB）
        try:
            # 语言代码映射
            nllb_lang_map = {
                'zh': 'zho_Hans', 'en': 'eng_Latn', 'ja': 'jpn_Jpan', 
                'ko': 'kor_Hang', 'de': 'deu_Latn', 'fr': 'fra_Latn',
                'es': 'spa_Latn', 'ru': 'rus_Cyrl', 'ar': 'arb_Arab'
            }
            m2m100_lang_map = {
                'zh': 'zh', 'en': 'en', 'ja': 'ja', 'ko': 'ko', 
                'de': 'de', 'fr': 'fr', 'es': 'es', 'ru': 'ru', 'ar': 'ar'
            }
            
            # 检测模型类型
            is_m2m100 = isinstance(TOKENIZER, M2M100Tokenizer) if M2M100Tokenizer else False
            
            if is_m2m100:
                # M2M100翻译流程
                src_code = m2m100_lang_map.get(source_lang, source_lang)
                tgt_code = m2m100_lang_map.get(tgt_lang, tgt_lang)
                TOKENIZER.src_lang = src_code
                inputs = TOKENIZER(translate_data['text'], return_tensors='pt')
                if torch and TRANSLATION_MODEL and torch.cuda.is_available() and next(TRANSLATION_MODEL.parameters()).is_cuda:
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                tgt_lang_id = TOKENIZER.get_lang_id(tgt_code)
                gen_tokens = TRANSLATION_MODEL.generate(
                    **inputs, 
                    forced_bos_token_id=tgt_lang_id,
                    max_length=512, 
                    num_beams=3,
                    early_stopping=True
                )
                decoded = TOKENIZER.batch_decode(gen_tokens, skip_special_tokens=True)
            else:
                # NLLB翻译流程
                src_code = nllb_lang_map.get(source_lang, source_lang)
                tgt_code = nllb_lang_map.get(tgt_lang, tgt_lang)
                TOKENIZER.src_lang = src_code
                inputs = TOKENIZER(translate_data['text'], return_tensors='pt')
                if torch and TRANSLATION_MODEL and torch.cuda.is_available() and next(TRANSLATION_MODEL.parameters()).is_cuda:
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                gen_tokens = TRANSLATION_MODEL.generate(
                    **inputs, 
                    forced_bos_token_id=TOKENIZER.convert_tokens_to_ids(tgt_code), 
                    max_length=512, 
                    num_beams=3
                )
                decoded = TOKENIZER.batch_decode(gen_tokens, skip_special_tokens=True)
            
            if not decoded:
                raise ValueError('Translation produced empty result')
            translated_text = decoded[0]
            
            return jsonify({'success': True, 'source_text': data.get('text'), 'translated_text': translated_text, 'source_language': source_lang, 'target_language': tgt_lang, 'segments': data.get('segments')})
        except Exception as e:
            import traceback
            print(f"[Translation Error in /process] {str(e)}")
            traceback.print_exc()
            # 失败后fallback到源文本
            return jsonify({'success': True, 'source_text': data.get('text'), 'translated_text': data.get('text'), 'source_language': source_lang, 'target_language': tgt_lang, 'segments': data.get('segments'), 'note': 'translation error, fallback to source'})
    # Fallback: return source only
    return jsonify({'success': True, 'source_text': data.get('text'), 'translated_text': data.get('text'), 'source_language': source_lang, 'target_language': tgt_lang, 'segments': data.get('segments'), 'note': 'translation fallback'})

if __name__ == '__main__':
    import argparse

    # Optional config.ini support (keeps CLI args as highest priority)
    try:
        from config_manager import config as app_config  # type: ignore
    except Exception:
        app_config = None

    default_host = getattr(app_config, 'service_host', '127.0.0.1')
    default_port = getattr(app_config, 'service_port', 50515)
    default_asr_model_size = getattr(app_config, 'asr_model_size', 'reazonspeech')
    default_translation_model = getattr(app_config, 'translation_model', 'facebook/nllb-200-1.3B')
    default_use_gpu = getattr(app_config, 'use_gpu', True)
    default_lazy_load = getattr(app_config, 'lazy_load_models', False)
    default_asr_transformers_chunk_sec = getattr(app_config, 'asr_transformers_chunk_sec', 30)
    default_asr_transformers_stride_sec = getattr(app_config, 'asr_transformers_stride_sec', 5.0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default=default_host)
    parser.add_argument('--port', default=default_port, type=int)
    parser.add_argument('--asr_model_size', default=default_asr_model_size)
    parser.add_argument('--translation_model', default=default_translation_model)  # 升级到1.3B（BLEU +8-10分）
    parser.add_argument('--no_gpu', action='store_true', help='强制使用CPU（忽略config.ini与CUDA检测）')
    parser.add_argument('--lazy-load', dest='lazy_load', action='store_true', default=default_lazy_load,
                        help='不要在启动时预加载模型，按需通过 /models/load 加载（适合显存紧张环境）')
    parser.add_argument('--eager-load', dest='lazy_load', action='store_false',
                        help='启动时预加载模型（覆盖 config.ini 的 lazy_load_models）')
    args = parser.parse_args()
    print('[PythonService] Starting with config:', args)
    print('[PythonService] HuggingFace endpoint:', os.environ.get('HF_ENDPOINT') or 'default')
    sys.stdout.flush()
    
    # 🔥 启动后自动加载模型
    if args.no_gpu:
        device = 'cpu'
        device_reason = '--no_gpu'
    elif not default_use_gpu:
        device = 'cpu'
        device_reason = 'config.ini use_gpu=false'
    elif _torch_cuda_available():
        device = 'cuda'
        device_reason = 'CUDA available'
    else:
        device = 'cpu'
        device_reason = 'CUDA not available'

    compute_type = 'float16' if device == 'cuda' else 'int8'
    if args.lazy_load:
        print(f'[PythonService] Lazy-load mode: ASR={args.asr_model_size}, Translation={args.translation_model}, Device={device}')
    else:
        print(f'[PythonService] Auto-loading models: ASR={args.asr_model_size}, Translation={args.translation_model}, Device={device}')
    print(f'[PythonService] Device selection reason: {device_reason}')
    sys.stdout.flush()

    # Expose defaults for /models/load (when payload omits fields)
    try:
        SERVICE_CONFIG.update({
            'asr_model_size': args.asr_model_size,
            'translation_model': args.translation_model,
            'use_gpu': bool(device == 'cuda'),
            'device': device,
            'compute_type': compute_type,
            'asr_transformers_chunk_sec': default_asr_transformers_chunk_sec,
            'asr_transformers_stride_sec': default_asr_transformers_stride_sec,
        })
    except Exception:
        pass
    
    if args.lazy_load:
        _set_status('idle', 0.0, 'Lazy-load enabled: models will be loaded on demand via /models/load.')
        print('[PythonService] Lazy-load enabled: not preloading models.')
        sys.stdout.flush()
    else:
        import threading

        def load_models():
            _load_models_async(args.asr_model_size, args.translation_model, device, compute_type)

        # 在后台线程加载模型，不阻塞Flask启动
        loader_thread = threading.Thread(target=load_models, daemon=True)
        loader_thread.start()
    
    app.run(host=args.host, port=args.port, debug=False)
