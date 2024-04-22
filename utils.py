import json
import os


# $                          READING PART                              $ #
# _______________________________________________________________________ #
def Read_File(filename: str) -> str:
    if not os.path.exists(path=filename):
        raise FileNotFoundError(f'File is not found: {filename}!')

    with open(file=filename, mode='r') as file:
        data = file.read()
        file.close()
    return data


def Read_Classes_From_Classification_File(filename: str) -> list[str]:
    data = Read_File(filename=filename)
    return [name for name in data.splitlines()]


def Read_Classes_With_ID_From_Json_File(filename: str) -> dict[str, int]:
    return json.loads(Read_File(filename=filename))
# _______________________________________________________________________ #
