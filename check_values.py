import pandas as pd
import numpy as np

FILE = "NF-UQ-NIDS-v2.csv"
CHUNK_SIZE = 200_000

NUMERIC_COLS = [
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "MIN_TTL",
    "MAX_TTL",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",
    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID",
    "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
]

mins = {c: np.inf for c in NUMERIC_COLS}
maxs = {c: -np.inf for c in NUMERIC_COLS}
nonfinite = {c: 0 for c in NUMERIC_COLS}
negative = {c: 0 for c in NUMERIC_COLS}

for chunk in pd.read_csv(FILE, chunksize=CHUNK_SIZE):

    for col in NUMERIC_COLS:
        x = pd.to_numeric(chunk[col], errors="coerce")

        mins[col] = min(mins[col], x.min())
        maxs[col] = max(maxs[col], x.max())

        nonfinite[col] += (~np.isfinite(x)).sum()

        # These are all non-negative flow/statistical quantities
        if col not in ["L4_SRC_PORT", "L4_DST_PORT"]:
            negative[col] += (x < 0).sum()

print("\n========== VALUE AUDIT ==========\n")

for col in NUMERIC_COLS:
    print(
        f"{col:35} "
        f"min={mins[col]:.6g} "
        f"max={maxs[col]:.6g} "
        f"nonfinite={nonfinite[col]} "
        f"negative={negative[col]}"
    )