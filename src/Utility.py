import socket
from contextlib import closing


CONFIG_SOURCE_DELIMITER = ';'


def dispatch_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def get_ip_port(url):
    ip_port = url[1].split(':')
    ip = ip_port[0]
    port = ip_port[1].split('/')[0]
    return ip, port


def valid_ip_address(ip):
    def is_ipv4(s):
        try:
            return str(int(s)) == s and 0 <= int(s) <= 255
        except (ValueError, Exception):
            return False

    if ip.count(".") == 3 and all(is_ipv4(i) for i in ip.split(".")):
        return True
    return False


def valid_port(port):
    try:
        if 1 <= int(port) <= 65535:
            return True
        else:
            raise ValueError
    except ValueError:
        return False


def extract_sources_from_config(path):
    sources = []
    with open(path, "r") as file:
        lines = file.readlines()
    if len(lines) > 0:
        for line in lines:
            source = line.split(CONFIG_SOURCE_DELIMITER)
            if len(source) == 2:
                source = [s.strip().replace(' ', '_').lower() for s in source]
                ip, port = get_ip_port(source)
                if len(source[0]) > 0:
                    sources.append(source)
    return sources
