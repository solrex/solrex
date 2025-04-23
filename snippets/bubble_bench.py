#!/bin/env python3
"""
批量对多服务器、多测试配置执行 sglang.bench_serving 压测，并根据压测结果绘制气泡图。

生成配置：
./bubble_bench.py -p > config.json

根据压测需求修改配置，然后执行压力测试（耗时较长，最好后台执行）：
./bubble_bench.py -c config.json &

压力测试报告将生成在 bubble_bench_report.html ，打开文件即是压力测试的 ECharts 图表。因为渲染图表需要
下载 echarts js，所以需要联网。

如果压测环境允许对不同的 endpoint 并行发压，可以传入并行发压参数 -j ：
./bubble_bench.py -c config.json -j 3 &

如果希望将多个压测配置在同一个目录下的压测结果统一生成报告，可以在执行完所有的压测后，对目录下所有文件生成报告：
（注意：所有 config*.json 中的 endpoints 不得同名，否则压测结果文件会被下一次压测清空导致丢失结果）

./bubble_bench.py -c config-1.json -j 3
./bubble_bench.py -c config-2.json -j 3
./bubble_bench.py -c config-3.json -j 3
./bubble_bench.py -g

在压测执行过程中，也可以随时通过 ./bubble_bench.py -g 生成基于当前压测结果的部分结果报告。
"""

import glob
import json
import os
import pandas as pd
import re
import subprocess
from argparse import ArgumentParser
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm

# 默认配置
DEFAULT_CONFIG_JSON = """{
    "sglang_bench_cmd": [
        "OPENAI_API_KEY=null python3 -m sglang.bench_serving --backend sglang-oai --disable-tqdm",
        "--dataset-path /workspace/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json",
        "--dataset-name random --random-range-ratio 1",
        "--random-input-len 2300 --random-output-len 700"
    ],
    // concurs：最大并发数：每个最大并发代表一次 benchmark
    // repeats：并发重复次数：与最大并发数相乘等于每次推理的总样本数。repeats 可以是单个数值，比如 10，或者一个列表。
    // 当它是一个列表时，列表大小必须与 "concurs" 完全一致，代表对应于每个并发的重复次数。
    "concurs": [ 1, 2, 4, 8,10,12,16,20,24,28,32,40,48,56,64,80,96,112,128],
    "repeats": [10,10,10,10,10, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 5,  5,  5],
    "endpoints": [
        {
            "base_url": "http://localhost:8080",
            // 吞吐的缩放系数：当计算多卡性能时，可通过该系数归一化到单卡性能，不提供时默认为 1.0
            "throughput_scale": 0.25,
            // 图例名：标识压测结果属于哪个曲线，可以不提供，默认使用简化后的 base_url
            "name": "TP4"
        },
        {
            "base_url": "http://localhost:8084",
            "throughput_scale": 0.5,
            "name": "TP2"
        },
        {
            "base_url": "http://localhost:8086",
            "throughput_scale": 1.0,
            "name": "TP1"
        }
    ]
}"""


def get_endpoint_name(endpoint):
    endpoint_name = (
        endpoint["name"]
        if "name" in endpoint and endpoint["name"] is not None
        else endpoint["base_url"]
    )
    endpoint_name = re.sub(r"^https?://", "", endpoint_name)
    endpoint_name = re.sub(r"[:]", "-", endpoint_name)
    return endpoint_name


def benchmark_one(sglang_bench_cmd, endpoint, concurrency, repeat, verbose=False):
    num_prompt = concurrency * repeat
    base_url = endpoint["base_url"]
    endpoint_name = get_endpoint_name(endpoint)
    throughput_scale = (
        endpoint["throughput_scale"] if "throughput_scale" in endpoint else 1.0
    )
    output_file = f"{endpoint_name}.bench"
    log_file = f"{endpoint_name}.log"
    tmp_file = f"{endpoint_name}-{concurrency}.bench"
    if os.path.isfile(tmp_file):
        os.remove(tmp_file)
    # Construct command
    cmd = " ".join(
        [
            *sglang_bench_cmd,
            "--base-url",
            base_url,
            "--request-rate",
            str(concurrency),
            "--max-concurrency",
            str(concurrency),
            "--num-prompt",
            str(num_prompt),
            "--output-file",
            tmp_file,
        ]
    )
    if verbose:
        print(f"{cmd}")
    # Execute command
    with open(log_file, "a") as f:
        subprocess.run(cmd, check=True, shell=True, stdout=f)
    with open(tmp_file, "r") as f:
        result = json.load(f)
    with open(output_file, "a") as f:
        result["_throughput_scale"] = throughput_scale
        result["_endpoint_name"] = endpoint_name
        json.dump(result, f)
        f.write("\n")
    if os.path.isfile(tmp_file):
        os.remove(tmp_file)


def benchmark_target(job_arg):
    job_id, sglang_bench_cmd, endpoint, concurrencies, repeats, verbose = job_arg
    endpoint_name = get_endpoint_name(endpoint)
    output_file = f"{endpoint_name}.bench"
    if os.path.isfile(output_file):
        os.remove(output_file)

    for i, concurrency in tqdm(
        enumerate(concurrencies),
        desc=f"Benching {endpoint_name}",
        total=len(concurrencies),
        position=job_id,
    ):
        repeat = repeats[i] if isinstance(repeats, list) else repeats
        benchmark_one(sglang_bench_cmd, endpoint, concurrency, repeat, verbose)


def run_benchmark(bench_config, parallel_jobs, verbose=False):
    if "repeats" in bench_config:
        repeats = bench_config["repeats"]
        if isinstance(repeats, list):
            assert len(bench_config["repeats"]) == len(
                bench_config["concurs"]
            ), "'repeats' list and 'concurs' list should have the same size"
    else:
        repeats = 10
    base_urls = set()
    names = set()
    for ep in bench_config["endpoints"]:
        assert ep["base_url"] not in base_urls, f"Duplicate endpoint.base_url {ep}"
        base_urls.add(ep["base_url"])
        if "name" in ep and ep["name"] is not None:
            assert ep["name"] not in names, f"Duplicate endpoint.name {ep}"
            names.add(ep["name"])

    print(
        "BubbleBenchmarking for endpoints: "
        + ", ".join([get_endpoint_name(v) for v in bench_config["endpoints"]])
    )
    jobs = [
        (
            job_id,
            bench_config["sglang_bench_cmd"],
            endpoint,
            bench_config["concurs"],
            repeats,
            verbose,
        )
        for job_id, endpoint in enumerate(bench_config["endpoints"])
    ]
    with Pool(processes=parallel_jobs) as pool:
        pool.map(benchmark_target, jobs)


def gen_report(bench_config=None):
    # 存储数据的字典
    data = defaultdict(list)
    bench_results = []

    if bench_config:
        for endpoint in bench_config["endpoints"]:
            endpoint_name = get_endpoint_name(endpoint)
            fname = f"{endpoint_name}.bench"
            if os.path.isfile(fname):
                bench_results.append(fname)
    else:
        bench_results = glob.glob("*.bench")

    # 读取并解析所有 JSON 数据
    data = []
    for file in bench_results:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    json_data = json.loads(line.strip())
                    data.append(json_data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line in {file}: {e}")

    # 转换为 DataFrame
    df = pd.DataFrame(data)

    # 确保所需字段存在
    required_fields = [
        'max_concurrency', '_endpoint_name', 'mean_itl_ms', 'request_throughput', '_throughput_scale',
        'median_itl_ms', 'p95_itl_ms', 'p99_itl_ms', 'std_itl_ms', 'mean_ttft_ms', 'median_ttft_ms',
        'p99_ttft_ms', 'std_ttft_ms', 'mean_e2e_latency_ms', 'median_e2e_latency_ms', 'p99_e2e_latency_ms',
        'input_throughput', 'output_throughput', 'concurrency', 'total_input_tokens', 'total_output_tokens'
    ]
    missing_fields = [field for field in required_fields if field not in df.columns]
    if missing_fields:
        raise ValueError(f"Missing required fields in data: {missing_fields}")

    # 获取所有唯一的分组和端点名称
    endpoints = df['_endpoint_name'].unique().tolist()
    max_concurrencies = [int(x) for x in sorted(df['max_concurrency'].unique().tolist())]

    # 生成 ECharts 配置
    charts = []

    # 计算 request_throughput * _throughput_scale 的范围
    throughput_scaled = df['request_throughput'] * df['_throughput_scale']
    throughput_scaled = throughput_scaled.replace([float('inf'), -float('inf')], float('nan')).dropna()
    if throughput_scaled.empty:
        raise ValueError("No valid throughput data after filtering")
    visual_map_min = max(throughput_scaled.min(), 1e-6)  # 避免 min 为 0
    visual_map_max = throughput_scaled.max()

    # 图表 1: mean_itl_ms 气泡图
    scatter_itl = {
        "title": {"text": "平均 ITL vs 并发", "left": "center"},
        'legend': {
            'top': 30,
            'data': endpoints,
        },
        'grid': {
            'top': 100,
            'left': 140
        },
        "tooltip": {"trigger": "item"},  # Formatter will be set in JavaScript
        "xAxis": {"name": "并发", "type": "value", "data": max_concurrencies},
        "yAxis": {"name": "平均 ITL (ms)", "type": "value"},
        "series": [],
        "visualMap": {
            'top': 30,
            "itemWidth": 25,
            'text': ['气泡大小\n(QPS/GPU)'],
            'textGap': 20,
            'calculable': True,
            'precision': 0.1,
            "inRange": {"symbolSize": [5, 70]},
            "min": visual_map_min,
            "max": visual_map_max,
            "dimension": 6,
        }
    }
    for endpoint in endpoints:
        endpoint_data = df[df['_endpoint_name'] == endpoint]
        series_data = [
            [
                row['max_concurrency'],
                row['mean_itl_ms'],
                row['median_itl_ms'],
                row['p95_itl_ms'],
                row['p99_itl_ms'],
                row['std_itl_ms'],
                row['request_throughput'] * row['_throughput_scale']
            ]
            for _, row in endpoint_data.iterrows()
        ]
        scatter_itl["series"].append({
            "name": endpoint,
            "type": "scatter",
            "data": series_data
        })
    charts.append(scatter_itl)

    # 图表 2: mean_ttft_ms 气泡图
    scatter_ttft = {
        "title": {"text": "平均 TTFT vs 并发", "left": "center"},
        'legend': {
            'top': 30,
            'data': endpoints,
        },
        'grid': {
            'top': 100,
            'left': 140
        },
        "tooltip": {"trigger": "item"},
        "xAxis": {"name": "并发", "type": "value", "data": max_concurrencies},
        "yAxis": {"name": "平均 TTFT (s)", "type": "value"},
        "series": [],
        "visualMap": {
            'top': 30,
            "itemWidth": 25,
            'text': ['气泡大小\n(QPS/GPU)'],
            'textGap': 20,
            'calculable': True,
            'precision': 0.1,
            "inRange": {"symbolSize": [5, 70]},
            "min": visual_map_min,
            "max": visual_map_max,
            "dimension": 5
        }
    }
    for endpoint in endpoints:
        endpoint_data = df[df['_endpoint_name'] == endpoint]
        series_data = [
            [
                row['max_concurrency'],
                row['mean_ttft_ms'] / 1000,
                row['median_ttft_ms'] / 1000,
                row['p99_ttft_ms'] / 1000,
                row['std_ttft_ms'] / 1000,
                row['request_throughput'] * row['_throughput_scale']
            ]
            for _, row in endpoint_data.iterrows()
        ]
        scatter_ttft["series"].append({
            "name": endpoint,
            "type": "scatter",
            "data": series_data
        })
    charts.append(scatter_ttft)

    # 图表 3: mean_e2e_latency_ms 气泡图
    scatter_e2e = {
        "title": {"text": "平均 E2E 延迟 vs 并发", "left": "center"},
        'legend': {
            'top': 30,
            'data': endpoints,
        },
        'grid': {
            'top': 100,
            'left': 140
        },
        "tooltip": {"trigger": "item"},
        "xAxis": {"name": "并发", "type": "value", "data": max_concurrencies},
        "yAxis": {"name": "平均 E2E 延迟 (s)", "type": "value"},
        "series": [],
        "visualMap": {
            'top': 30,
            "itemWidth": 25,
            'text': ['气泡大小\n(QPS/GPU)'],
            'textGap': 20,
            'calculable': True,
            'precision': 0.1,
            "inRange": {"symbolSize": [5, 70]},
            "min": visual_map_min,
            "max": visual_map_max,
            "dimension": 4
        }
    }
    for endpoint in endpoints:
        endpoint_data = df[df['_endpoint_name'] == endpoint]
        series_data = [
            [
                row['max_concurrency'],
                row['mean_e2e_latency_ms'] / 1000,
                row['median_e2e_latency_ms'] / 1000,
                row['p99_e2e_latency_ms'] / 1000,
                row['request_throughput'] * row['_throughput_scale']
            ]
            for _, row in endpoint_data.iterrows()
        ]
        scatter_e2e["series"].append({
            "name": endpoint,
            "type": "scatter",
            "data": series_data
        })
    charts.append(scatter_e2e)

    # 其他折线图
    metrics = [
        ("request_throughput", "QPS", "吞吐(QPS)"),
        ("input_throughput", "输入Token吞吐", "吞吐(token/s)"),
        ("output_throughput", "生成Token吞吐", "吞吐(token/s)"),
        ("concurrency", "服务端并发", "服务端并发"),
        ("total_io_throughput", "总Token吞吐", "吞吐(token/s)"),
        ("total_input_tokens", "总输入Token数", "Tokens"),
        ("total_output_tokens", "总生成Token数", "Tokens")
    ]

    for metric, title, y_axis_name in metrics:
        line_chart = {
            "title": {"text": f"{title} vs 并发", "left": "center"},
            'legend': {
                'top': 30,
                'data': endpoints,
            },
            'grid': {
                'top': 60,
                'left': 140
            },
            "tooltip": {"trigger": "axis"},
            "xAxis": {"name": "并发", "type": "value", "data": max_concurrencies},
            "yAxis": {"name": y_axis_name, "type": "value"},
            "series": []
        }
        for endpoint in endpoints:
            endpoint_data = df[df['_endpoint_name'] == endpoint].sort_values('max_concurrency')
            if metric == "total_io_throughput":
                values = (endpoint_data['input_throughput'] + endpoint_data['output_throughput']).tolist()
            else:
                values = endpoint_data[metric].tolist()
            series_data = [[int(row['max_concurrency']), round(float(row['value']), 2)] for _, row in endpoint_data[['max_concurrency']].assign(value=values).iterrows()]
            line_chart["series"].append({
                "name": endpoint,
                "type": "line",
                "data": series_data
            })
        charts.append(line_chart)

    # 定义 JavaScript formatter 函数
    formatter_js = """
    const formatters = [
        // Formatter for 平均 ITL vs 并发
        function(params) {
            var v = params.value;
            return `
                Endpoint: ${params.seriesName}<br/>
                并发: ${v[0].toFixed(2)}<br/>
                平均 ITL: ${v[1].toFixed(2)} ms<br/>
                中位数 ITL: ${v[2].toFixed(2)} ms<br/>
                P95 ITL: ${v[3].toFixed(2)} ms<br/>
                P99 ITL: ${v[4].toFixed(2)} ms<br/>
                标准差 ITL: ${v[5].toFixed(2)} ms<br/>
                QPS/GPU: ${v[6].toFixed(2)}
            `;
        },
        // Formatter for 平均 TTFT vs 并发
        function(params) {
            var v = params.value;
            return `
                Endpoint: ${params.seriesName}<br/>
                并发: ${v[0].toFixed(2)}<br/>
                平均 TTFT: ${v[1].toFixed(2)} s<br/>
                中位数 TTFT: ${v[2].toFixed(2)} s<br/>
                P99 TTFT: ${v[3].toFixed(2)} s<br/>
                标准差 TTFT: ${v[4].toFixed(2)} s<br/>
                QPS/GPU: ${v[5].toFixed(2)}
            `;
        },
        // Formatter for 平均 E2E 延迟 vs 并发
        function(params) {
            var v = params.value;
            return `
                Endpoint: ${params.seriesName}<br/>
                并发: ${v[0].toFixed(2)}<br/>
                平均 E2E 延迟: ${v[1].toFixed(2)} s<br/>
                中位数 E2E 延迟: ${v[2].toFixed(2)} s<br/>
                P99 E2E 延迟: ${v[3].toFixed(2)} s<br/>
                QPS/GPU: ${v[4].toFixed(2)}
            `;
        }
    ];
    """

    # 生成 HTML 文件
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Bubble Bench Charts</title>
        <script src="https://fastly.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
        <style>
            body {{ margin-right: 15px; padding: 0;}}
            .chart {{ width: 100%; height: 400px; margin-bottom: 20px; box-sizing: border-box; }}
        </style>
    </head>
    <body>
        <div id="charts"></div>
        <script>
            {formatter_js}
            const charts = {json.dumps(charts, indent=2, ensure_ascii=False)};
            const chartDoms = [];
            const chartContainer = document.getElementById('charts');

            // 创建图表容器
            charts.forEach((_, index) => {{
                const div = document.createElement('div');
                div.id = `chart${{index}}`;
                div.className = 'chart';
                chartContainer.appendChild(div);
                chartDoms.push(echarts.init(div, null, {{renderer: 'canvas'}}));
            }});

            // 配置图表
            charts.forEach((option, index) => {{
                if (index < 3 && formatters[index]) {{
                    option.tooltip.formatter = formatters[index];
                }}
                if (option.visualMap) {{
                    option.visualMap.formatter = function (v) {{
                        return v.toFixed(2);
                    }};
                }}
                option.toolbox = {{ feature: {{ saveAsImage: {{}} }} }};
                option.dataZoom = [
                    {{ type: 'slider', xAxisIndex: 0, filterMode: 'none' }},
                    {{ 
                        type: 'inside', 
                        xAxisIndex: 0, 
                        filterMode: 'none',
                        zoomOnMouseWheel: false, // 禁用滚轮缩放
                        moveOnMouseWheel: false  // 禁用滚轮平移
                    }}
                ];
                chartDoms[index].setOption(option);
            }});

            // 防抖函数
            function debounce(fn, delay) {{
                let timeout;
                return function(...args) {{
                    clearTimeout(timeout);
                    timeout = setTimeout(() => fn.apply(this, args), delay);
                }};
            }}

            // 高亮联动
            chartDoms.forEach(chart => {{
                chart.on('mouseover', debounce(function(param) {{
                    chartDoms.forEach(c => {{
                        if (c !== chart) {{
                            c.dispatchAction({{
                                type: 'highlight',
                                seriesIndex: param.seriesIndex,
                                dataIndex: param.dataIndex
                            }});
                        }}
                    }});
                }}, 100));
                chart.on('mouseout', debounce(function(param) {{
                    chartDoms.forEach(c => {{
                        c.dispatchAction({{
                            type: 'downplay',
                            seriesIndex: param.seriesIndex,
                            dataIndex: param.dataIndex
                        }});
                    }});
                }}, 100));
            }});

            // dataZoom 同步
            chartDoms.forEach((chart, index) => {{
                chart.on('dataZoom', debounce(function(param) {{
                    const option = chart.getOption();
                    const dataZoom = option.dataZoom[0]; // 获取 slider 的 dataZoom 配置
                    chartDoms.forEach((otherChart, otherIndex) => {{
                        if (otherIndex !== index) {{
                            otherChart.setOption({{
                                dataZoom: [
                                    {{ start: dataZoom.start, end: dataZoom.end }},
                                    {{ start: dataZoom.start, end: dataZoom.end }}
                                ]
                            }});
                        }}
                    }});
                }}, 100));
            }});
        </script>
    </body>
    </html>
    """

    # 将 HTML 写入文件
    with open("bubble_bench_report.html", "w") as f:
        f.write(html_content)
    print("BubbleBenchmarking report generated: bubble_bench_report.html")


def main(args):
    config_str = DEFAULT_CONFIG_JSON
    if args.print_config:
        print(config_str)
        return 0
    elif args.gen_report:
        return gen_report()

    if args.config:
        with open(args.config, "r") as f:
            config_str = f.read()
    # 过滤掉整行注释
    json_str = re.sub(r"(?m)^\s*//.*\n", "", config_str)
    config = json.loads(json_str)

    run_benchmark(config, args.jobs, args.verbose)
    gen_report(config)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Benchmark multiple service, generate bubble chart to compare throughput, TPOT and TTFT"
    )
    parser.add_argument(
        "--config", "-c", type=str, help="Benchmark config file in json format."
    )
    parser.add_argument(
        "--print-config",
        "-p",
        action="store_true",
        help="Print config template for subsequent editing.",
    )
    parser.add_argument(
        "--gen-report",
        "-g",
        action="store_true",
        help="Generate report for all .json files in current directory.",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of endpoint benchmarking jobs to run simultaneously.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose information",
    )
    args = parser.parse_args()
    main(args)
