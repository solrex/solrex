#!/bin/env python3
"""
批量对多服务器、多测试配置执行 sglang.bench_serving 压测，并根据压测结果绘制气泡图。

生成配置：
./bubble_bench.py --print-config > config.json

根据压测需求修改配置，然后执行压力测试（耗时较长，最好后台执行）：
./bubble_bench.py -c config.json &

压力测试报告将生成在 bubble_bench_report.html ，打开文件即是压力测试的 ECharts 图表。

如果压测环境允许对不同的 endpoint 并行发压，可以传入并行发压参数 -j ：
./bubble_bench.py -c config.json -j 3 &

如果希望将多个压测配置在同一个目录下的压测结果统一生成报告，可以在执行完所有的压测后，对目录下所有文件生成报告：
（注意：所有 config*.json 中的 endpoints 不得同名，否则压测结果文件会被下一次压测清空导致丢失结果）
./bubble_bench.py -c config-1.json -j 3
./bubble_bench.py -c config-2.json -j 3
./bubble_bench.py -c config-3.json -j 3
./bubble_bench.py --gen-report
"""

import glob
import json
import os
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
    endpoint_name = re.sub(r"[:-]", ".", endpoint_name)
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
        result["_endpoing_name"] = endpoint_name
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
            for concurrency in bench_config["concurs"]:
                fname = f"{endpoint_name}.bench"
                if os.path.isfile(fname):
                    bench_results.append(fname)
    else:
        bench_results = glob.glob("*.bench")

    # 解析 JSON 文件并提取所需字段
    for file in bench_results:
        try:
            with open(file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        json_data = json.loads(line)
                        endpoint_name = json_data["_endpoing_name"]
                        throughput_scale = json_data["_throughput_scale"]
                        concur = json_data["max_concurrency"]
                        qps = json_data["request_throughput"] * throughput_scale
                        tpot = json_data["mean_tpot_ms"]
                        ttft = json_data["mean_ttft_ms"]
                        data[endpoint_name].append((concur, tpot, ttft, qps))
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file}")
            continue

    # 按并发数降序
    for endpoint_name in data:
        data[endpoint_name].sort(key=lambda x: x[0])

    endpoints = sorted(data.keys())
    concur_values = [r for endpoint_name in data for r, _, _, _ in data[endpoint_name]]
    tpot_values = [t for endpoint_name in data for _, t, _, _ in data[endpoint_name]]
    ttft_values = [t for endpoint_name in data for _, _, t, _ in data[endpoint_name]]
    qps_values = [q for endpoint_name in data for _, _, _, q in data[endpoint_name]]

    concur_max = max(concur_values)
    ttft_min = min(ttft_values, default=0)
    ttft_max = max(ttft_values)
    qps_min = min(qps_values, default=0)
    qps_max = max(qps_values)

    # 定义 schema
    schema = [
        {"name": "并发", "index": 0, "text": "请求并发"},
        {"name": "tpot", "index": 1, "text": "Mean TPOT (ms)"},
        {"name": "ttft", "index": 2, "text": "Mean TTFT (ms)"},
        {"name": "qps", "index": 3, "text": "每卡吞吐 (req/s/GPU)"},
    ]

    # 生成 ECharts HTML 代码
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Performance Bubble Chart</title>
        <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
    </head>
    <body>
        <div id="chart" style="width: 100%; height: 600px;"></div>
        <script type="text/javascript">
            var chartDom = document.getElementById('chart');
            var myChart = echarts.init(chartDom);
            var option;

            const schema = {json.dumps(schema)};
            const itemStyle = {{
                opacity: 0.8,
                shadowBlur: 10,
                shadowOffsetX: 0,
                shadowOffsetY: 0,
                shadowColor: 'rgba(0,0,0,0.3)'
            }};

            option = {{
                legend: {{
                    top: 10,
                    data: {endpoints},
                    textStyle: {{ fontSize: 16 }}
                }},
                grid: {{
                    left: '10%',
                    right: 150,
                    top: '18%',
                    bottom: '10%'
                }},
                tooltip: {{
                    backgroundColor: 'rgba(255,255,255,0.7)',
                    formatter: function (param) {{
                        var value = param.value;
                        return '<div style="border-bottom: 1px solid rgba(255,255,255,.3); font-size: 18px;padding-bottom: 7px;margin-bottom: 7px">'
                            + param.seriesName + ' Request Rate ' + value[0] + '</div>'
                            + schema[1].text + ': ' + value[1].toFixed(2) + '<br>'
                            + schema[2].text + ': ' + value[2].toFixed(2) + '<br>'
                            + schema[3].text + ': ' + value[3].toFixed(2) + '<br>';
                    }}
                }},
                xAxis: {{
                    type: 'value',
                    name: '并发',
                    nameGap: 16,
                    nameTextStyle: {{ fontSize: 16 }},
                    max: {concur_max},
                    splitLine: {{ show: false }}
                }},
                yAxis: {{
                    type: 'value',
                    name: 'Mean TPOT (ms)',
                    nameLocation: 'end',
                    nameGap: 20,
                    nameTextStyle: {{ fontSize: 16 }},
                    splitLine: {{ show: false }}
                }},
                visualMap: [
                    {{
                        left: 'right',
                        top: '10%',
                        dimension: 3,
                        min: {qps_min},
                        max: {qps_max},
                        itemWidth: 30,
                        itemHeight: 120,
                        calculable: true,
                        precision: 0.1,
                        text: ['气泡大小: 每卡吞吐 req/s/GPU'],
                        textGap: 30,
                        inRange: {{ symbolSize: [10, 70] }},
                        outOfRange: {{ symbolSize: [10, 70], color: ['rgba(255,255,255,0.4)'] }},
                        controller: {{ inRange: {{ color: ['#c23531'] }}, outOfRange: {{ color: ['#999'] }} }}
                    }},
                    {{
                        left: 'right',
                        bottom: '5%',
                        dimension: 2,
                        min: {ttft_min},
                        max: {ttft_max},
                        itemHeight: 120,
                        text: ['颜色深浅: Mean TTFT'],
                        textGap: 30,
                        inRange: {{ colorLightness: [0.9, 0.5] }},
                        outOfRange: {{ color: ['rgba(255,255,255,0.4)'] }},
                        controller: {{ inRange: {{ color: ['#c23531'] }}, outOfRange: {{ color: ['#999'] }} }}
                    }}
                ],
                series: [
                    {','.join([
                        f"{{name: '{endpoint_name}', type: 'scatter', itemStyle: itemStyle, " +
                        "data: [" +
                        ','.join([
                            f"[{concur}, {tpot:.2f}, {ttft}, {qps:.2f}]"
                            for concur, tpot, ttft, qps in sorted(data[endpoint_name], key=lambda x: x[0])
                        ]) +
                        "]}"
                        for endpoint_name in endpoints
                    ])}
                ]
            }};
            option && myChart.setOption(option);
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
        action="store_true",
        help="Print config template for subsequent editing.",
    )
    parser.add_argument(
        "--gen-report",
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
