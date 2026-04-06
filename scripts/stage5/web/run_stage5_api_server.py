"""
ToT Tree Data API Server
提供 ToT 节点数据的 RESTful API
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

# 绝对路径
PROJECT_ROOT = Path(r'D:\001_softwares\VS Code Demo\cpo_project')
RUNS_ROOT = PROJECT_ROOT / 'outputs' / 'eval' / 'runs'

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
        
        # 静态文件
        return super().do_GET()

    def send_json(self, data):
        """发送 JSON 响应"""
        response = json.dumps(data, ensure_ascii=False)
        response_bytes = response.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', len(response_bytes))
        self.end_headers()
        self.wfile.write(response_bytes)

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
        print("[HINT] 若使用浏览器访问，建议优先使用 http://127.0.0.1:<port>/tot_viewer.html")
        return

    handler_cls = partial(ToTAPIHandler, directory=str(web_dir))
    server_address = (host, port)
    httpd = HTTPServer(server_address, handler_cls)
    print(f"ToT API Server running at http://{host}:{port}")
    print(f"Serving files from: {web_dir}")
    print(f"Viewer URL: http://{host}:{port}/tot_viewer.html")
    print(f"Performance URL: http://{host}:{port}/performance_viewer.html")

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
