import os
import json
import sys
import inspect


class local_error(Exception):
    pass


def get_script_dir(follow_symlinks=True):
    '''получить директорию со исполняемым скриптом'''
    if getattr(sys, 'frozen', False):  # type: ignore
        path = os.path.abspath(sys.executable)
    else:
        path = inspect.getabsfile(get_script_dir)
    if follow_symlinks:
        path = os.path.realpath(path)
    return os.path.dirname(path)


def read(file_name, folder_name):
    path = os.path.join(get_script_dir(), folder_name, file_name)

    if not os.path.exists(path):
        raise local_error("Папка с файлом не существует")

    data = []
    try:
        with open(path, 'r', encoding="utf-8") as f:
            data = json.load(f)
    except json.decoder.JSONDecodeError:
        raise local_error("Некорректное чтение файла")

    if not isinstance(data, dict):
        raise local_error("Некорректное чтение файла")

    return data


def write(data, gen_mode):

    path = os.path.join(get_script_dir(), "graphics")

    if not os.path.exists(path):
        os.mkdir(path)

    path_to_file = path + f"/{gen_mode}_{len(data)}"
    with open(path_to_file, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_file_list():
    path = os.path.join(get_script_dir(), "graphics")

    if not os.path.exists(path):
        raise local_error("Папка graphics не существует")

    return os.listdir(path)
