import os
import sys
import time
import importlib
import datetime
import logging


try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except Exception:
    term_width = 100

TOTAL_BAR_LENGTH = 10.
last_time = time.time()
begin_time = last_time

# +
KST = datetime.timezone(datetime.timedelta(hours=9))
CURRENT_TIME = datetime.datetime.now(KST).strftime('%y_%m_%d_%H:%M:%S')

# this is test

def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append(' %d/%d ' % (current + 1, total))
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 10):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def dict_str(config, indent=0):
    msg = ""
    if not isinstance(config, dict):
        return ''
    for value, item in config.items():
        if isinstance(item, dict):
            msg += f'{"  "*indent}{value:<15}: \n' + dict_str(item, indent + 1)
        else:
            msg += f'{"  "*indent}{value:<15}: {item}\n'
    return msg


def get_logger(name, logging_dir):
    logger = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] >>> %(message)s')
    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(os.path.join(logging_dir, 'logger.log'))
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=logging.DEBUG)
    return logger


def get_yaml(path):
    import yaml, re
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(path) as f:
        config_yml = yaml.load(f, loader)
    return config_yml


def print_log(name, dct, size=50, sw=None, epoch=0):
    """
        print in pretty format
    """
    dump = int((size - len(name)) / 2 - 2)
    buff = 0 if (size - len(name)) / 2 == 0 else 1
    msg = (f'{"=" * dump}  {name}  {"=" * (dump + buff)}\n')
    for value, item in dct.items():
        msg += (f'| {value:<{int(size/2)-2}} {item:>{int(size/2)-2}} |\n')

        if sw is not None:
            sw.add_scalar(value, item, epoch)

    msg += (f'{"=" * size}')
    print(msg)


def update_config(yml, argv):
    assert yml is not None

    for name, val in argv.__dict__.items():
        if val is None:
            continue

        if name in yml.keys():
            yml[name] = val
        else:
            for key in yml.keys():
                if isinstance(yml[key], dict):
                    update_config(yml[key], argv)
            yml[name] = val

def set_log_dir(path):
    from datetime import datetime
    now = datetime.now()
    t_msg = f"_{now.year}_{now.month}_{now.day}_{now.hour}:{now.minute}:{now.second}"
    path += t_msg

    os.makedirs(path, exist_ok=True)
    print(f"log path is {path}")

    return path
