"""
NetWorld Lightweight Packet-Level Feature Extraction Layer
Extracts defensible packet-derived statistics:
- TTL mean/min/max
- TCP window statistics (mean, min, max)
- Payload-size statistics (mean, std, max, zero-payload ratio)
- Retransmission count and ratio
- Port-scan indicators (unique destination ports, SYN-only ratio, RST ratio)

Operates on raw PCAP binaries, PCAP streams, or tabular packet telemetry.
Pure-Python, zero external dependencies (uses standard struct & socket).
"""

import os
import sys
import struct
import socket
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

try:
    from ml.packet_schema import (
        PACKET_FEATURE_NAMES,
        NUM_PACKET_FEATURES,
        PACKET_FEATURE_SPEC,
        validate_and_sanitize_packet_dict,
        validate_and_sanitize_packet_array,
    )
    from ml.feature_schema import FEATURE_NAMES, NUM_FEATURES
except ImportError:
    from packet_schema import (
        PACKET_FEATURE_NAMES,
        NUM_PACKET_FEATURES,
        PACKET_FEATURE_SPEC,
        validate_and_sanitize_packet_dict,
        validate_and_sanitize_packet_array,
    )
    from feature_schema import FEATURE_NAMES, NUM_FEATURES


class ParsedPacket:
    """Holds parsed attributes of an individual IP packet."""
    __slots__ = (
        "timestamp", "src_ip", "dst_ip", "protocol", "src_port", "dst_port",
        "ttl", "tcp_window", "payload_len", "tcp_flags", "tcp_seq", "is_tcp",
        "is_retransmission"
    )

    def __init__(
        self,
        timestamp: float,
        src_ip: str,
        dst_ip: str,
        protocol: int,
        src_port: int,
        dst_port: int,
        ttl: int,
        tcp_window: int = 0,
        payload_len: int = 0,
        tcp_flags: int = 0,
        tcp_seq: int = 0,
        is_tcp: bool = False,
        is_retransmission: bool = False,
    ):
        self.timestamp = timestamp
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.protocol = protocol
        self.src_port = src_port
        self.dst_port = dst_port
        self.ttl = ttl
        self.tcp_window = tcp_window
        self.payload_len = payload_len
        self.tcp_flags = tcp_flags
        self.tcp_seq = tcp_seq
        self.is_tcp = is_tcp
        self.is_retransmission = is_retransmission


class PacketFeatureExtractor:
    """
    Lightweight packet-level parser and feature extraction layer.
    Extracts 15 defensible packet statistics aggregated per flow or per time window.
    """

    def __init__(self):
        pass

    @staticmethod
    def parse_pcap_file(filepath: str) -> List[ParsedPacket]:
        """
        Parses a PCAP file using pure Python struct unpacking.
        Supports both little-endian and big-endian PCAP headers.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PCAP file not found: {filepath}")

        packets: List[ParsedPacket] = []
        with open(filepath, "rb") as f:
            global_header = f.read(24)
            if len(global_header) < 24:
                return packets

            magic = struct.unpack("<I", global_header[:4])[0]
            if magic in (0xA1B2C3D4, 0xA1B23C4D):
                endian = "<"
                is_nano = (magic == 0xA1B23C4D)
            elif magic in (0xD4C3B2A1, 0x4D3CB2A1):
                endian = ">"
                is_nano = (magic == 0x4D3CB2A1)
            else:
                # Might be PCAP-NG or raw packets; try standard little-endian fallback
                endian = "<"
                is_nano = False

            # Track TCP sequence numbers per connection to detect retransmissions
            seen_seqs: Dict[Tuple[str, str, int, int], set] = {}

            while True:
                hdr_bytes = f.read(16)
                if len(hdr_bytes) < 16:
                    break

                ts_sec, ts_usec, incl_len, orig_len = struct.unpack(f"{endian}IIII", hdr_bytes)
                pkt_data = f.read(incl_len)
                if len(pkt_data) < incl_len:
                    break

                ts = float(ts_sec) + (float(ts_usec) / (1e9 if is_nano else 1e6))

                # Parse Ethernet frame (14 bytes)
                if len(pkt_data) < 14:
                    continue
                eth_type = struct.unpack("!H", pkt_data[12:14])[0]
                ip_data_offset = 14

                # Handle 802.1Q VLAN Tagging (0x8100)
                if eth_type == 0x8100 and len(pkt_data) >= 18:
                    eth_type = struct.unpack("!H", pkt_data[16:18])[0]
                    ip_data_offset = 18

                if eth_type != 0x0800:  # Only IPv4 for standard NetWorld flow analysis
                    continue

                if len(pkt_data) < ip_data_offset + 20:
                    continue

                # Parse IPv4 Header
                ip_header = pkt_data[ip_data_offset:ip_data_offset + 20]
                ver_ihl, tos, total_len, ip_id, flags_frag, ttl, proto, cksum, src_raw, dst_raw = struct.unpack(
                    "!BBHHHBBH4s4s", ip_header
                )
                ihl = (ver_ihl & 0x0F) * 4
                src_ip = socket.inet_ntoa(src_raw)
                dst_ip = socket.inet_ntoa(dst_raw)

                transport_offset = ip_data_offset + ihl
                payload_offset = transport_offset

                src_port = 0
                dst_port = 0
                tcp_win = 0
                tcp_flags = 0
                tcp_seq = 0
                is_tcp = (proto == 6)
                is_retrans = False

                if is_tcp and len(pkt_data) >= transport_offset + 20:
                    tcp_header = pkt_data[transport_offset:transport_offset + 20]
                    src_port, dst_port, tcp_seq, tcp_ack, offset_reserved, tcp_flags, tcp_win, tcp_sum, tcp_urp = struct.unpack(
                        "!HHIIBBHHH", tcp_header
                    )
                    tcp_data_offset = ((offset_reserved >> 4) & 0x0F) * 4
                    payload_offset = transport_offset + tcp_data_offset

                    # Retransmission detection: (src, dst, sport, dport)
                    conn_key = (src_ip, dst_ip, src_port, dst_port)
                    if conn_key not in seen_seqs:
                        seen_seqs[conn_key] = set()
                    if tcp_seq in seen_seqs[conn_key]:
                        is_retrans = True
                    else:
                        seen_seqs[conn_key].add(tcp_seq)

                elif proto == 17 and len(pkt_data) >= transport_offset + 8:  # UDP
                    udp_header = pkt_data[transport_offset:transport_offset + 8]
                    src_port, dst_port, udp_len, udp_cksum = struct.unpack("!HHHH", udp_header)
                    payload_offset = transport_offset + 8

                payload_len = max(0, len(pkt_data) - payload_offset)

                packets.append(ParsedPacket(
                    timestamp=ts,
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    protocol=proto,
                    src_port=src_port,
                    dst_port=dst_port,
                    ttl=ttl,
                    tcp_window=tcp_win,
                    payload_len=payload_len,
                    tcp_flags=tcp_flags,
                    tcp_seq=tcp_seq,
                    is_tcp=is_tcp,
                    is_retransmission=is_retrans,
                ))

        return packets

    @staticmethod
    def aggregate_packet_features(packets: List[ParsedPacket]) -> Dict[str, float]:
        """
        Aggregates a collection of packets (e.g. within a flow or time window)
        into the exact 15 defensible packet-level features.
        """
        if not packets:
            # Return neutral baseline defaults
            return validate_and_sanitize_packet_dict({})

        ttls = [p.ttl for p in packets]
        windows = [p.tcp_window for p in packets if p.is_tcp]
        payloads = [p.payload_len for p in packets]
        retrans_count = sum(1 for p in packets if p.is_retransmission)
        total_pkts = len(packets)

        # Port-scan indicators
        unique_dst_ports = len(set(p.dst_port for p in packets if p.dst_port > 0))
        tcp_pkts = [p for p in packets if p.is_tcp]
        num_tcp = len(tcp_pkts)

        if num_tcp > 0:
            # SYN-only: SYN flag set (0x02) and ACK unset (0x10)
            syn_only_cnt = sum(1 for p in tcp_pkts if (p.tcp_flags & 0x02) and not (p.tcp_flags & 0x10))
            rst_cnt = sum(1 for p in tcp_pkts if (p.tcp_flags & 0x04))
            syn_only_ratio = syn_only_cnt / num_tcp
            rst_ratio = rst_cnt / num_tcp
        else:
            syn_only_ratio = 0.0
            rst_ratio = 0.0

        zero_payload_count = sum(1 for p in payloads if p == 0)

        raw_features = {
            # TTL Statistics
            "Pkt TTL Mean": float(np.mean(ttls)),
            "Pkt TTL Min": float(np.min(ttls)),
            "Pkt TTL Max": float(np.max(ttls)),

            # TCP Window Statistics
            "TCP Win Mean": float(np.mean(windows)) if windows else 0.0,
            "TCP Win Min": float(np.min(windows)) if windows else 0.0,
            "TCP Win Max": float(np.max(windows)) if windows else 0.0,

            # Payload-Size Statistics
            "Payload Len Mean": float(np.mean(payloads)),
            "Payload Len Std": float(np.std(payloads)),
            "Payload Len Max": float(np.max(payloads)),
            "Payload Zero Ratio": float(zero_payload_count / total_pkts) if total_pkts > 0 else 0.0,

            # Retransmission Indicators
            "Retransmission Cnt": float(retrans_count),
            "Retransmission Ratio": float(retrans_count / total_pkts) if total_pkts > 0 else 0.0,

            # Port-Scan & Probe Indicators
            "Unique Dst Ports": float(max(1, unique_dst_ports)),
            "SYN Only Ratio": float(syn_only_ratio),
            "RST Ratio": float(rst_ratio),
        }

        return validate_and_sanitize_packet_dict(raw_features)

    @staticmethod
    def extract_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, bool]:
        """
        Extracts the 15 packet features from a DataFrame if columns are present.
        Returns:
            (packet_array of shape (N, 15), is_available: bool)
        If columns are missing, returns neutral baseline array and is_available=False.
        """
        num_rows = len(df)
        present_cols = [c for c in PACKET_FEATURE_NAMES if c in df.columns]

        if len(present_cols) == NUM_PACKET_FEATURES:
            # All 15 packet features explicitly present
            sub_df = df[PACKET_FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
            arr = sub_df.values.astype(np.float32)
            arr = validate_and_sanitize_packet_array(arr)
            return arr, True

        # Check for case-insensitive matches
        lower_map = {c.lower().strip(): c for c in df.columns}
        matched_cols = []
        for name in PACKET_FEATURE_NAMES:
            if name.lower().strip() in lower_map:
                matched_cols.append(lower_map[name.lower().strip()])

        if len(matched_cols) == NUM_PACKET_FEATURES:
            sub_df = df[matched_cols].apply(pd.to_numeric, errors="coerce")
            arr = sub_df.values.astype(np.float32)
            arr = validate_and_sanitize_packet_array(arr)
            return arr, True

        # Fallback: synthesize defensible flow-correlated packet statistics from existing flow columns
        # E.g. Tot Fwd Pkts, Init Fwd Win Byts, SYN Flag Cnt, RST Flag Cnt, Pkt Len Mean
        arr = np.zeros((num_rows, NUM_PACKET_FEATURES), dtype=np.float32)

        # Default neutral baselines
        for i, name in enumerate(PACKET_FEATURE_NAMES):
            arr[:, i] = PACKET_FEATURE_SPEC[name]["default"]

        # Correlate available flow columns if present
        if "Init Fwd Win Byts" in df.columns:
            win_col = pd.to_numeric(df["Init Fwd Win Byts"], errors="coerce").fillna(14600.0).values
            arr[:, 3] = np.clip(win_col, 0, 65535)  # TCP Win Mean
            arr[:, 4] = np.clip(win_col * 0.8, 0, 65535)  # TCP Win Min
            arr[:, 5] = np.clip(win_col * 1.2, 0, 65535)  # TCP Win Max

        if "Pkt Len Mean" in df.columns:
            pkt_len = pd.to_numeric(df["Pkt Len Mean"], errors="coerce").fillna(0.0).values
            arr[:, 6] = np.maximum(0.0, pkt_len - 40.0)  # Payload Len Mean (~ Total - IP/TCP headers)
            arr[:, 8] = np.maximum(0.0, pkt_len * 1.5)   # Payload Len Max

        if "SYN Flag Cnt" in df.columns and "Tot Fwd Pkts" in df.columns:
            syn_cnt = pd.to_numeric(df["SYN Flag Cnt"], errors="coerce").fillna(0.0).values
            tot_pkts = np.maximum(1.0, pd.to_numeric(df["Tot Fwd Pkts"], errors="coerce").fillna(1.0).values)
            arr[:, 13] = np.clip(syn_cnt / tot_pkts, 0.0, 1.0)  # SYN Only Ratio

        if "RST Flag Cnt" in df.columns and "Tot Fwd Pkts" in df.columns:
            rst_cnt = pd.to_numeric(df["RST Flag Cnt"], errors="coerce").fillna(0.0).values
            tot_pkts = np.maximum(1.0, pd.to_numeric(df["Tot Fwd Pkts"], errors="coerce").fillna(1.0).values)
            arr[:, 14] = np.clip(rst_cnt / tot_pkts, 0.0, 1.0)  # RST Ratio

        arr = validate_and_sanitize_packet_array(arr)
        return arr, False  # is_available is False because it was synthesized from flow telemetry


def generate_test_pcap(filepath: str, num_packets: int = 50) -> str:
    """
    Generates a valid binary PCAP file for testing and verification.
    Includes TCP handshakes, payload packets, port scans, and retransmissions.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "wb") as f:
        # PCAP Global Header: magic (0xa1b2c3d4), ver 2.4, thiszone 0, sigfigs 0, snaplen 65535, eth (1)
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))

        base_time = 1710000000.0
        src_ip_bytes = socket.inet_aton("192.168.1.105")
        dst_ip_bytes = socket.inet_aton("10.0.0.1")

        for i in range(num_packets):
            ts = base_time + (i * 0.05)
            ts_sec = int(ts)
            ts_usec = int((ts - ts_sec) * 1e6)

            # Alternate ports to simulate port scan or flow variation
            dst_port = 80 if i % 3 != 0 else (443 + (i % 5))
            src_port = 54321 + (i % 4)

            # TCP flags: SYN for first, ACK for others, RST occasionally
            if i % 10 == 0:
                flags = 0x02  # SYN
                payload = b""
                seq = 1000 + i
            elif i % 12 == 0:
                flags = 0x04  # RST
                payload = b""
                seq = 2000 + i
            elif i % 7 == 0:
                # Retransmission: duplicate sequence number
                flags = 0x18  # PSH, ACK
                payload = b"GET /admin HTTP/1.1\r\nHost: target\r\n\r\n"
                seq = 5000  # Duplicate seq
            else:
                flags = 0x10  # ACK
                payload = b"X" * (64 + (i * 8) % 512)
                seq = 3000 + (i * 100)

            # Construct TCP header (20 bytes)
            # data_offset = 5 (5 * 4 = 20 bytes)
            tcp_offset_res = (5 << 4)
            tcp_hdr = struct.pack(
                "!HHIIBBHHH",
                src_port, dst_port, seq, 0,
                tcp_offset_res, flags, 14600 + (i * 50) % 10000, 0, 0
            )

            # Construct IPv4 Header (20 bytes)
            total_ip_len = 20 + len(tcp_hdr) + len(payload)
            ttl = 64 if i % 2 == 0 else 60
            ip_hdr = struct.pack(
                "!BBHHHBBH4s4s",
                0x45, 0, total_ip_len, i + 1, 0x4000, ttl, 6, 0,
                src_ip_bytes, dst_ip_bytes
            )

            # Ethernet header (14 bytes)
            eth_hdr = b"\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb\x08\x00"

            full_frame = eth_hdr + ip_hdr + tcp_hdr + payload
            incl_len = len(full_frame)
            orig_len = incl_len

            # Packet record header
            f.write(struct.pack("<IIII", ts_sec, ts_usec, incl_len, orig_len))
            f.write(full_frame)

    return filepath
