import threading
import queue

import requests

q = queue.Queue()
valid_proxies = [] 

with open("proxy_list.txt", "r") as f:
    proxies = f.readlines().split("\n")
    for proxy in proxies:
        q.put(proxy)


def check_proxies():
    global q
    while not q.empty():
        proxy = q.get()
        try:
            r = requests.get("https://www.google.com", proxies={"http": proxy, "https": proxy}, timeout=5)

        except:
            continue
        if r.status_code == 200:
            print(proxy)

for _ in range(10):
    t = threading.Thread(target=check_proxies).start()