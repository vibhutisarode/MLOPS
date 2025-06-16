import sys
from src.mlproject.logger import logging


def error_message_detail(error, error_detail: sys):
    """
    Returns detailed error message with traceback information.
    If no active exception (i.e., sys.exc_info() returns None), fallback to basic message.
    """
    exc_type, exc_value, exc_tb = sys.exc_info()

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        return f"Error occurred in script: [{file_name}] at line number: [{exc_tb.tb_lineno}] error message: [{str(error)}]"
    else:
        return f"Error message: [{str(error)}]"


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
