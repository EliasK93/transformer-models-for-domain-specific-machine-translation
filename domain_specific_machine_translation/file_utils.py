import json


def read_txt(path: str) -> list[str]:
    """
    Reads txt file and returns it as list of strings, one string per line.

    :param path: path to the file to read
    :return: list of strings, one string per line
    """
    with open(path, encoding="utf-8") as f:
        return [line.rstrip() for line in f.readlines()]


def write_txt(lines: list[str], path: str):
    """
    Writes a list of strings and write it to a txt file, one string per line.

    :param lines: string list to write to file
    :param path: path to the file to write
    """
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line+"\n")


def write_jsonl(dict_: dict, path: str):
    """
    Writes a dicts values to JSONL format file, ignoring the keys.

    :param dict_: dictionary to write to file
    :param path: path to the file to write
    """
    with open(path, 'w') as f:
        for entry in dict_.values():
            json.dump(entry, f)
            f.write('\n')
