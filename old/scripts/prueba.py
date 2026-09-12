import asyncio
import multiprocessing
import queue
import time

async def poll(q):
    i = 0
    while True:
        print("First", i)
        i += 1
        await asyncio.sleep(1)
        print("Second", i)
        i += 1
        await asyncio.sleep(1)
        if i == 10:
            q.put([True])
            print("terminado")
        
async def stop():
    q = multiprocessing.Queue()
    poller = asyncio.ensure_future(poll(q))
    while True:
        await asyncio.sleep(0.1)
        try:
            cola = q.get(0)
            if cola[0] == True:
                poller.cancel()
                return
        except queue.Empty:
            ...

asyncio.run(stop())