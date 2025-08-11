from enum import Enum

from pandas import DataFrame


class Classes(Enum):
    TEST = "test"
    TRAIN = "train"
    VALID = "valid"


class DfData:
    """
    Classes used to process and carry data
    """

    def __init__(self, df: DataFrame, file: str):
        """
        initialization
        :param df: df data, store df data
        :param file: The path where the df data is stored in a file in a certain format, that is, the file storage path
        """
        self.df = df
        self.file = file

    def __str__(self):
        return f"df: {self.df}, file: {self.file}"


class OpcodeData:

    def __init__(self) -> None:
        self.func_name = ""
        self.docstring = ""
        self.code = ""
        self.opcode = ""
        self.line = []
