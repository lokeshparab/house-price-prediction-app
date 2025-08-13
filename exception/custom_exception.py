import sys, traceback
from logger.custom_logger import CustomLogger

logging = CustomLogger().get_logger(__file__)

class CustomException(Exception):
    """Custom exception class that captures the stack trace and provides a formatted error message."""
    def __init__(self, error_message: str,error_details:sys):
        # super().__init__(error_message)
        _,_,exc_tb = sys.exc_info()
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.lineno = exc_tb.tb_lineno
        self.error_message = str(error_message)

        # self.traceback_str = "".join(traceback.format_exception(*error_details.exc_info()))

    def __str__(self):
        return (
            f"Error in [{self.file_name}] at line [{self.lineno}]\n"
            f"Message : {self.error_message}\n"
            # f"Traceback:\n{self.traceback_str}"
        )

if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        app_exc = CustomException(e, sys)
        # logging.error(str(app_exc))
        raise app_exc

    # try:
    #     a = 1/0
    # except Exception as e:
    #     raise e
