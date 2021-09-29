import queue
import threading
import torch
import logging

save_queue = None
print_queue = False


def save_file_worker():
    global save_queue
    while True:
        sd, fn = save_queue.get()
        try:
            if print_queue:
                logging.info(f'Doing: {fn}')
            torch.save(sd, fn)
        except Exception as e:
            logging.exception(str(e))
        save_queue.task_done()


def fast_save(sd, fn):
    global save_queue
    if save_queue is None:
        logging.warning('save_queue did not init, please call init_save_queue')
        raise RuntimeError('save_queue did not init')

    save_queue.put((sd, fn))


def join_save_queue():
    global save_queue, print_queue
    if save_queue is None:
        logging.warning('save_queue did not init, please call init_save_queue')
        raise RuntimeError('save_queue did not init')

    logging.info(f'Remaining Save Tasks: {save_queue.qsize()}')
    print_queue = True
    save_queue.join()


def init_save_queue():
    global save_queue

    save_queue = queue.Queue()
    t = threading.Thread(target=save_file_worker)
    t.daemon = True
    t.start()
