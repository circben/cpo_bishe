"""
ToT Tree Data API Server
提供 ToT 节点数据的 RESTful API
http://127.0.0.1:8888/web_html/index.html
"""
import json
import os
import argparse
import socket
import subprocess
import sys
from functools import partial
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import urllib.request
import urllib.error

# 绝对路径
PROJECT_ROOT = Path(r'D:\001_softwares\VS Code Demo\cpo_project')
RUNS_ROOT = PROJECT_ROOT / 'outputs' / 'eval' / 'runs'
NODES_ROOT = PROJECT_ROOT / 'outputs' / 'eval' / 'nodes'
REMOTE_API_BASE = os.getenv("REMOTE_API_BASE", "http://127.0.0.1:8088").rstrip("/")
LOCAL_JOBS_ROOT = PROJECT_ROOT / 'outputs' / 'remote_inference' / 'jobs'
LOCAL_SYNC_SCRIPT = os.getenv("LOCAL_SYNC_SCRIPT", "scripts/local_sync/rsync_pull_wsl.ps1")
LOCAL_SYNC_ENABLED = os.getenv("LOCAL_SYNC_ENABLED", "1").strip() in {"1", "true", "yes"}

# 成功路径文件
SUCCESS_PATH_FILES = {
    'gsm8k': PROJECT_ROOT / 'data' / 'interim' / 'success_paths' / 'gsm8k_success_paths_sft.jsonl',
    'strategyqa': PROJECT_ROOT / 'data' / 'interim' / 'success_paths' / 'strategyqa_success_paths_sft.jsonl',
}

# DPO 偏好数据文件
DPO_FILES = {
    'gsm8k': PROJECT_ROOT / 'data' / 'processed' / 'dpo' / 'gsm8k' / 'gsm8k_train.jsonl',
    'strategyqa': PROJECT_ROOT / 'data' / 'processed' / 'dpo' / 'strategyqa' / 'strategyqa_train.jsonl',
}

DATASET_SOURCES = {
    'gsm8k': {
        'name': 'gsm8k',
        'model': 'Qwen3-8B_444',
        'dir': PROJECT_ROOT / 'data' / 'interim' / 'tot_nodes' / 'gsm8k' / 'Qwen3-8B_444',
    },
    'strategyqa': {
        'name': 'strategyqa',
        'model': 'Qwen3-8B_433',
        'dir': PROJECT_ROOT / 'data' / 'interim' / 'tot_nodes' / 'strategyqa' / 'Qwen3-8B_433',
    },
}


def _to_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class ToTAPIHandler(SimpleHTTPRequestHandler):
    """自定义 HTTP 处理类"""
    _cache = {}

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        query = urllib.parse.parse_qs(parsed_path.query)
        
        # Debug
        print(f"Request: {path}")

        if path == '/api/datasets' or path == '/api/datasets/':
            self.send_json(self.get_datasets())
            return

        if path.startswith('/api/remote/status/'):
            job_id = urllib.parse.unquote(path[len('/api/remote/status/'):])
            self._proxy_json('GET', f'/status/{job_id}')
            return

        if path.startswith('/api/remote/logs/'):
            job_id = urllib.parse.unquote(path[len('/api/remote/logs/'):])
            self._proxy_text('GET', f'/logs/{job_id}')
            return

        if path.startswith('/api/remote/result/'):
            job_id = urllib.parse.unquote(path[len('/api/remote/result/'):])
            self._proxy_json('GET', f'/result/{job_id}')
            return

        if path.startswith('/api/local/result/'):
            job_id = urllib.parse.unquote(path[len('/api/local/result/'):])
            self._send_local_result(job_id)
            return

        if path.startswith('/api/local/logs/'):
            job_id = urllib.parse.unquote(path[len('/api/local/logs/'):])
            self._send_local_logs(job_id)
            return

        if path == '/api/performance/runs' or path == '/api/performance/runs/':
            dataset = query.get('dataset', ['all'])[0].strip().lower()
            limit = _to_int(query.get('limit', [50])[0], 50)
            self.send_json(self.get_performance_runs(dataset=dataset, limit=limit))
            return
        
        # API: 获取文件列表
        if path == '/api/files' or path == '/api/files/':
            dataset = query.get('dataset', [''])[0].strip()
            page = _to_int(query.get('page', [1])[0], 1)
            page_size = _to_int(query.get('page_size', [100])[0], 100)
            result = self.get_file_list(dataset=dataset, page=page, page_size=page_size)
            print(f"Files count: {len(result.get('files', []))}")
            self.send_json(result)
            return

        if path.startswith('/api/generate/'):
            filename = urllib.parse.unquote(path[len('/api/generate/'):])
            self.send_json(self.generate_visualization(filename))
            return
        
        # API: 获取单个文件内容
        if path.startswith('/api/tree/'):
            filename = path[10:]  # 去掉 '/api/tree/'
            filename = urllib.parse.unquote(filename)
            self.send_json(self.get_tree_data(filename))
            return

        # API: 获取成功路径统计
        if path == '/api/stats/success-paths' or path == '/api/stats/success-paths/':
            self.send_json(self.get_success_paths_stats())
            return

        # API: 获取 DPO 偏好数据统计
        if path == '/api/stats/dpo-pairs' or path == '/api/stats/dpo-pairs/':
            self.send_json(self.get_dpo_pairs_stats())
            return

        # API: 获取 ToT 节点批次列表
        if path == '/api/nodes/batches' or path == '/api/nodes/batches/':
            self.send_json(self.get_nodes_batches())
            return

        # API: 获取批次中的样本列表
        if path.startswith('/api/nodes/samples/'):
            batch_path = urllib.parse.unquote(path[len('/api/nodes/samples/'):])
            self.send_json(self.get_nodes_samples(batch_path))
            return

        # API: 获取单个样本数据
        if path.startswith('/api/nodes/sample/'):
            sample_path = urllib.parse.unquote(path[len('/api/nodes/sample/'):])
            self.send_json(self.get_node_sample(sample_path))
            return

        # 静态文件
        return super().do_GET()

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        if path == '/api/remote/infer' or path == '/api/remote/infer/':
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length) if length else b''
            try:
                payload = json.loads(raw.decode('utf-8')) if raw else {}
            except json.JSONDecodeError:
                self.send_json({'error': 'Invalid JSON'}, status=400)
                return
            self._proxy_json('POST', '/infer', payload)
            return

        if path == '/api/local/sync' or path == '/api/local/sync/':
            if not LOCAL_SYNC_ENABLED:
                self.send_json({'error': 'Local sync disabled'}, status=400)
                return
            ok, detail = self._run_local_sync()
            if not ok:
                self.send_json(detail or {'error': 'Local sync failed'}, status=500)
                return
            self.send_json({'ok': True})
            return

        self.send_json({'error': 'Not Found'}, status=404)

    def send_json(self, data, status=200):
        """发送 JSON 响应"""
        response = json.dumps(data, ensure_ascii=False)
        response_bytes = response.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', len(response_bytes))
        self.end_headers()
        self.wfile.write(response_bytes)

    def send_text(self, text, status=200, content_type='text/plain; charset=utf-8'):
        response_bytes = text.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', len(response_bytes))
        self.end_headers()
        self.wfile.write(response_bytes)

    def _remote_url(self, path):
        return f"{REMOTE_API_BASE}{path}"

    def _proxy_json(self, method, path, payload=None):
        url = self._remote_url(path)
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            headers['Content-Type'] = 'application/json; charset=utf-8'
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode('utf-8')
                self.send_json(json.loads(body), status=resp.status)
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode('utf-8') if exc.fp else ''
            self.send_json({'error': err_body or str(exc)}, status=exc.code)
        except Exception as exc:
            self.send_json({'error': str(exc)}, status=500)

    def _proxy_text(self, method, path):
        url = self._remote_url(path)
        req = urllib.request.Request(url, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode('utf-8')
                self.send_text(body, status=resp.status)
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode('utf-8') if exc.fp else ''
            self.send_text(err_body or str(exc), status=exc.code)
        except Exception as exc:
            self.send_text(str(exc), status=500)

    def _local_job_dir(self, job_id: str) -> Path:
        return LOCAL_JOBS_ROOT / job_id

    def _send_local_result(self, job_id: str) -> None:
        result_path = self._local_job_dir(job_id) / 'output' / 'result.json'
        if not result_path.exists():
            self.send_json({'error': f'Local result not found: {job_id}'}, status=404)
            return
        self.send_json(json.loads(result_path.read_text(encoding='utf-8')))

    def _send_local_logs(self, job_id: str) -> None:
        log_path = self._local_job_dir(job_id) / 'run.log'
        if not log_path.exists():
            self.send_text(f'Local log not found: {job_id}', status=404)
            return
        self.send_text(log_path.read_text(encoding='utf-8'))

    def _run_local_sync(self):
        script_path = Path(LOCAL_SYNC_SCRIPT)
        if not script_path.exists():
            return False, {'error': f'Local sync script not found: {script_path}'}
        try:
            result = subprocess.run(
                [
                    'powershell',
                    '-ExecutionPolicy',
                    'Bypass',
                    '-File',
                    str(script_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return False, {
                    'error': 'Local sync failed',
                    'stderr': (result.stderr or '').strip(),
                    'stdout': (result.stdout or '').strip(),
                }
            return True, None
        except Exception as exc:
            return False, {'error': str(exc)}

    def _scan_dataset_files(self, dataset_id):
        if dataset_id in self._cache:
            return self._cache[dataset_id]

        config = DATASET_SOURCES.get(dataset_id)
        if not config:
            return []

        model_dir = config['dir']
        print(f"Scanning: {model_dir}, exists: {model_dir.exists()}")
        files = []

        if model_dir.exists():
            for json_file in sorted(model_dir.glob('*.json')):
                # 跳过 .meta.json 文件
                if json_file.name.endswith('.meta.json'):
                    continue
                
                try:
                    # 读取 meta 文件获取问题
                    question = ''
                    gold = ''
                    qid = ''
                    
                    meta_file = json_file.with_suffix('.meta.json')
                    if meta_file.exists():
                        with open(meta_file, 'r', encoding='utf-8') as mf:
                            meta = json.load(mf)
                            question = meta.get('question', '')
                            gold = meta.get('gold', '')
                            qid = meta.get('qid', json_file.stem)
                    
                    # 提取 qid 最后5位用于排序
                    qid_short = ''
                    if qid:
                        qid_short = qid[-5:] if len(qid) >= 5 else qid
                    
                    files.append({
                        'name': json_file.name,
                        'path': json_file.relative_to(PROJECT_ROOT).as_posix(),
                        'model': config['model'],
                        'dataset': dataset_id,
                        'question': question,
                        'gold': gold,
                        'qid': qid,
                        'qid_short': qid_short,
                        'sort_key': int(qid_short) if qid_short.isdigit() else 0
                    })
                except Exception as e:
                    print(f"Error reading {json_file}: {e}")
                    continue
        
        files.sort(key=lambda x: (x['sort_key'], x['qid']))
        self._cache[dataset_id] = files
        return files

    def get_datasets(self):
        datasets = []
        for dataset_id, cfg in DATASET_SOURCES.items():
            files = self._scan_dataset_files(dataset_id)
            datasets.append({
                'id': dataset_id,
                'name': cfg['name'],
                'model': cfg['model'],
                'count': len(files),
            })
        return {'datasets': datasets}

    def get_file_list(self, dataset='', page=1, page_size=100):
        """按数据集分页获取 ToT 树文件列表"""
        datasets_info = self.get_datasets()['datasets']
        valid_dataset_ids = {d['id'] for d in datasets_info}

        if dataset not in valid_dataset_ids:
            dataset = 'gsm8k' if 'gsm8k' in valid_dataset_ids else next(iter(valid_dataset_ids), '')

        all_dataset_files = self._scan_dataset_files(dataset) if dataset else []

        page = max(page, 1)
        page_size = max(min(page_size, 500), 1)
        total = len(all_dataset_files)
        total_pages = max((total + page_size - 1) // page_size, 1)
        page = min(page, total_pages)

        start = (page - 1) * page_size
        end = start + page_size
        paged_files = all_dataset_files[start:end]

        return {
            'datasets': datasets_info,
            'dataset': dataset,
            'page': page,
            'page_size': page_size,
            'total': total,
            'total_pages': total_pages,
            'files': paged_files,
        }

    def generate_visualization(self, filename):
        """调用脚本生成单题 Mermaid 可视化文件"""
        try:
            file_path = (PROJECT_ROOT / filename).resolve()
            root_path = PROJECT_ROOT.resolve()
            if root_path not in file_path.parents and file_path != root_path:
                return {'error': 'Invalid file path'}
            if not file_path.exists():
                return {'error': f'File not found: {filename}'}

            rel = file_path.relative_to(PROJECT_ROOT)
            parts = rel.parts
            if len(parts) >= 6:
                dataset = parts[3]
                model = parts[4]
            else:
                dataset = 'unknown'
                model = 'unknown'

            out_dir = PROJECT_ROOT / 'outputs' / 'visualization' / 'web_mermaid' / dataset / model
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f'{file_path.stem}.mmd'

            script_path = PROJECT_ROOT / 'scripts' / 'analysis' / 'visualize_tot_tree.py'
            cmd = [
                sys.executable,
                str(script_path),
                '--tree-json',
                str(file_path),
                '--out',
                str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode != 0:
                return {
                    'error': 'Failed to generate mermaid visualization',
                    'stderr': proc.stderr.strip(),
                    'stdout': proc.stdout.strip(),
                }

            return {
                'ok': True,
                'output_path': out_path.relative_to(PROJECT_ROOT).as_posix(),
                'stdout': proc.stdout.strip(),
            }
        except Exception as e:
            return {'error': str(e)}

    def get_tree_data(self, filename):
        """获取单个 ToT 树数据"""
        file_path = PROJECT_ROOT / filename
        
        if not file_path.exists():
            return {'error': f'File not found: {filename}'}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取 meta 信息
            meta_file = file_path.with_suffix('.meta.json')
            meta = {}
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            
            return {
                'nodes': data,
                'meta': meta
            }
        except Exception as e:
            return {'error': str(e)}

    def _safe_rel_path(self, path_obj):
        try:
            return path_obj.relative_to(PROJECT_ROOT).as_posix()
        except Exception:
            return path_obj.as_posix()

    def _collect_run_json_files(self):
        if not RUNS_ROOT.exists():
            return []
        files = []
        for json_file in RUNS_ROOT.rglob('*.json'):
            if json_file.name.endswith('.meta.json'):
                continue
            files.append(json_file)
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files

    def get_success_paths_stats(self):
        """统计成功路径数据"""
        results = []
        total_count = 0
        total_prompt_len = 0
        total_response_len = 0

        for dataset_id, file_path in SUCCESS_PATH_FILES.items():
            if not file_path.exists():
                results.append({
                    'dataset': dataset_id,
                    'count': 0,
                    'avg_prompt_len': 0,
                    'avg_response_len': 0,
                    'error': f'File not found: {file_path}'
                })
                continue

            try:
                count = 0
                prompt_len_sum = 0
                response_len_sum = 0

                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            prompt_len_sum += len(data.get('prompt', '') or '')
                            response_len_sum += len(data.get('response', '') or '')
                            count += 1
                        except json.JSONDecodeError:
                            continue

                avg_prompt = round(prompt_len_sum / count) if count > 0 else 0
                avg_response = round(response_len_sum / count) if count > 0 else 0

                results.append({
                    'dataset': dataset_id,
                    'count': count,
                    'avg_prompt_len': avg_prompt,
                    'avg_response_len': avg_response,
                })

                total_count += count
                total_prompt_len += prompt_len_sum
                total_response_len += response_len_sum

            except Exception as e:
                results.append({
                    'dataset': dataset_id,
                    'count': 0,
                    'avg_prompt_len': 0,
                    'avg_response_len': 0,
                    'error': str(e)
                })

        return {
            'total_count': total_count,
            'total_avg_prompt_len': round(total_prompt_len / total_count) if total_count > 0 else 0,
            'total_avg_response_len': round(total_response_len / total_count) if total_count > 0 else 0,
            'datasets': results
        }

    def get_dpo_pairs_stats(self):
        """统计 DPO 偏好数据"""
        results = []

        for dataset_id, file_path in DPO_FILES.items():
            if not file_path.exists():
                results.append({
                    'dataset': dataset_id,
                    'pairs': 0,
                    'questions': 0,
                    'error': f'File not found: {file_path}'
                })
                continue

            try:
                count = 0
                questions = set()
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            count += 1
                            # 从prompt中提取Question部分
                            prompt = data.get('prompt', '')
                            q_start = prompt.find('Question:')
                            q_end = prompt.find('?', q_start)
                            if q_start != -1 and q_end != -1:
                                question = prompt[q_start:q_end+1]
                                questions.add(question)
                        except json.JSONDecodeError:
                            continue

                results.append({
                    'dataset': dataset_id,
                    'pairs': count,
                    'questions': len(questions),
                })

            except Exception as e:
                results.append({
                    'dataset': dataset_id,
                    'pairs': 0,
                    'questions': 0,
                    'error': str(e)
                })

        total_pairs = sum(r.get('pairs', 0) for r in results)
        total_questions = sum(r.get('questions', 0) for r in results)

        return {
            'total_pairs': total_pairs,
            'total_questions': total_questions,
            'datasets': results
        }

    def get_nodes_batches(self):
        """获取 outputs/eval/nodes 下的所有批次列表"""
        batches = []
        if not NODES_ROOT.exists():
            return {'batches': batches, 'error': f'Nodes root not found: {NODES_ROOT}'}

        try:
            # 遍历 dataset/split/run_name
            for dataset_dir in sorted(NODES_ROOT.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                dataset_name = dataset_dir.name

                for split_dir in sorted(dataset_dir.iterdir()):
                    if not split_dir.is_dir():
                        continue
                    split_name = split_dir.name

                    for run_dir in sorted(split_dir.iterdir()):
                        if not run_dir.is_dir():
                            continue
                        batch_name = run_dir.name

                        batches.append({
                            'dataset': dataset_name,
                            'split': split_name,
                            'batch_name': batch_name,
                            'path': f'{dataset_name}/{split_name}/{batch_name}'
                        })
        except Exception as e:
            return {'batches': batches, 'error': str(e)}

        return {'batches': batches}

    def get_nodes_samples(self, batch_path):
        """获取指定批次中的样本列表，按方法分组"""
        # batch_path 格式: dataset/split/batch_name
        parts = batch_path.split('/')
        if len(parts) != 3:
            return {'error': f'Invalid batch path: {batch_path}', 'samples_by_method': {}, 'methods': []}

        dataset, split, batch_name = parts
        batch_dir = NODES_ROOT / dataset / split / batch_name

        if not batch_dir.exists():
            return {'error': f'Batch not found: {batch_path}', 'samples_by_method': {}, 'methods': []}

        samples_by_method = {}
        try:
            for method_dir in sorted(batch_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                method_name = method_dir.name
                samples = []

                for sample_file in sorted(method_dir.glob('sample_*.json')):
                    if sample_file.name.endswith('.meta.json'):
                        continue
                    try:
                        with open(sample_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        samples.append({
                            'file': sample_file.name,
                            'index': data.get('index', 0),
                            'correct': data.get('correct', False),
                            'prediction': data.get('prediction', ''),
                        })
                    except Exception:
                        continue

                if samples:
                    samples_by_method[method_name] = samples

        except Exception as e:
            return {'error': str(e), 'samples_by_method': {}, 'methods': []}

        methods = sorted(samples_by_method.keys())
        return {
            'samples_by_method': samples_by_method,
            'methods': methods
        }

    def get_node_sample(self, sample_path):
        """获取单个样本的完整数据"""
        # sample_path 格式: dataset/split/batch_name/method/sample_xxx.json
        parts = sample_path.split('/')
        if len(parts) != 5:
            return {'error': f'Invalid sample path: {sample_path}', 'data': None}

        file_path = NODES_ROOT / sample_path
        if not file_path.exists():
            return {'error': f'Sample not found: {sample_path}', 'data': None}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {'data': data}
        except Exception as e:
            return {'error': str(e), 'data': None}

    def get_performance_runs(self, dataset='all', limit=50):
        """读取 outputs/eval/runs 下评估结果并生成性能对比聚合。"""
        limit = max(1, min(limit, 500))
        run_files = self._collect_run_json_files()

        runs = []
        method_metrics = {}
        datasets = set()

        for run_file in run_files:
            if len(runs) >= limit:
                break
            try:
                with open(run_file, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
            except Exception:
                continue

            task = str(payload.get('task', '')).strip().lower()
            if dataset not in ('', 'all') and task != dataset:
                continue

            summaries = payload.get('summaries', [])
            if not isinstance(summaries, list):
                summaries = []

            datasets.add(task or 'unknown')
            methods = []

            for item in summaries:
                method = str(item.get('method', 'unknown'))
                accuracy = item.get('accuracy', 0.0)
                latency = item.get('avg_latency_sec', 0.0)
                count = item.get('count', 0)

                methods.append({
                    'method': method,
                    'accuracy': accuracy,
                    'avg_latency_sec': latency,
                    'count': count,
                })

                if method not in method_metrics:
                    method_metrics[method] = {
                        'method': method,
                        'runs': 0,
                        'accuracy_sum': 0.0,
                        'latency_sum': 0.0,
                        'best_accuracy': 0.0,
                    }

                metric = method_metrics[method]
                metric['runs'] += 1
                metric['accuracy_sum'] += float(accuracy or 0.0)
                metric['latency_sum'] += float(latency or 0.0)
                metric['best_accuracy'] = max(metric['best_accuracy'], float(accuracy or 0.0))

            run_item = {
                'task': payload.get('task', 'unknown'),
                'split': payload.get('split', ''),
                'created_at': payload.get('created_at', ''),
                'run_name': run_file.parents[1].name if len(run_file.parents) >= 2 else run_file.parent.name,
                'file_path': self._safe_rel_path(run_file),
                'methods': methods,
            }
            runs.append(run_item)

        method_summary = []
        for method_name, item in sorted(method_metrics.items()):
            run_count = item['runs']
            avg_acc = (item['accuracy_sum'] / run_count) if run_count else 0.0
            avg_latency = (item['latency_sum'] / run_count) if run_count else 0.0
            method_summary.append({
                'method': method_name,
                'runs': run_count,
                'avg_accuracy': avg_acc,
                'best_accuracy': item['best_accuracy'],
                'avg_latency_sec': avg_latency,
            })

        return {
            'root': self._safe_rel_path(RUNS_ROOT),
            'dataset': dataset,
            'datasets': sorted([d for d in datasets if d]),
            'run_count': len(runs),
            'method_summary': method_summary,
            'runs': runs,
        }


def check_port_available(host, port):
    """检查端口是否可用，避免被其他服务占用导致请求异常。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def run_server(host='127.0.0.1', port=8888):
    """启动 API 服务器"""
    web_dir = Path(__file__).parent

    if not check_port_available(host, port):
        print(f"[ERROR] {host}:{port} 已被占用，服务器未启动。")
        print("[HINT] 请先关闭占用进程，或使用 --port 指定新端口。")
        print("[HINT] 若使用浏览器访问，建议优先使用 http://127.0.0.1:<port>/web_html/index.html")
        return

    handler_cls = partial(ToTAPIHandler, directory=str(web_dir))
    server_address = (host, port)
    httpd = HTTPServer(server_address, handler_cls)
    print(f"ToT API Server running at http://{host}:{port}")
    print(f"Serving files from: {web_dir}")
    print(f"Viewer URL: http://{host}:{port}/web_html/index.html")
    print(f"  - ToT Viewer: http://{host}:{port}/web_html/tot_viewer.html")
    print(f"  - Performance: http://{host}:{port}/web_html/performance_viewer.html")
    print(f"  - Statistics: http://{host}:{port}/web_html/statistics_report.html")

    httpd.serve_forever()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ToT Tree Data API Server')
    parser.add_argument('--host', default='127.0.0.1', help='监听地址，默认 127.0.0.1')
    parser.add_argument('--port', type=int, default=8888, help='监听端口，默认 8888')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_server(host=args.host, port=args.port)
