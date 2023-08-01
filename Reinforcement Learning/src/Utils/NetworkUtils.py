import socket


''' Finds an open port for Socket to Run on'''
def find_open_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 0))
    addr, port = sock.getsockname()
    sock.close()
    return port
