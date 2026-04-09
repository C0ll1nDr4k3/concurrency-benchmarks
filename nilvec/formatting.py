from colorama import Fore, Style


def format_benchmark_header(name):
    if name in {"Redis", "Weaviate", "USearch"} or "FAISS" in name:
        name_color = Fore.MAGENTA
    else:
        name_color = Fore.CYAN
    return (
        f"\n{Style.BRIGHT}{Fore.CYAN}Benchmarking Throughput{Style.RESET_ALL}: "
        f"{Style.BRIGHT}{name_color}{name}{Style.RESET_ALL}"
    )


def format_throughput_line(
    num_threads,
    num_insert_threads,
    num_search_threads,
    throughput,
    prev,
    build_time=None,
    search_latencies=None,
    insert_latencies=None,
):
    if prev is None:
        throughput_color = Fore.CYAN
    elif throughput >= prev * 1.05:
        throughput_color = Fore.GREEN
    elif throughput <= prev * 0.95:
        throughput_color = Fore.RED
    else:
        throughput_color = Fore.YELLOW
    build_str = ""
    if build_time is not None:
        build_str = (
            f"{Fore.BLUE}Build:{Style.RESET_ALL} "
            f"{Fore.WHITE}{build_time:.2f}s{Style.RESET_ALL} | "
        )
    lat_str = ""
    if search_latencies is not None:
        s_p50, s_p95, s_p99 = search_latencies
        lat_str += (
            f" | {Fore.BLUE}R p50/p99:{Style.RESET_ALL} "
            f"{Fore.GREEN}{s_p50:.1f}/{s_p99:.1f}ms{Style.RESET_ALL}"
        )
    if insert_latencies is not None:
        i_p50, i_p95, i_p99 = insert_latencies
        lat_str += (
            f" | {Fore.BLUE}W p50/p99:{Style.RESET_ALL} "
            f"{Fore.GREEN}{i_p50:.1f}/{i_p99:.1f}ms{Style.RESET_ALL}"
        )
    return (
        f"  {Fore.BLUE}Threads:{Style.RESET_ALL} {num_threads} "
        f"{Fore.WHITE}(W={num_insert_threads}, R={num_search_threads}){Style.RESET_ALL} -> "
        f"{build_str}"
        f"{Fore.BLUE}Throughput:{Style.RESET_ALL} "
        f"{Style.BRIGHT}{throughput_color}{throughput:.0f}{Style.RESET_ALL} ops/s"
        f"{lat_str}"
    )


def _format_elapsed(seconds):
    """Format elapsed seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{int(hours)}h {int(minutes)}m {secs:.0f}s"
