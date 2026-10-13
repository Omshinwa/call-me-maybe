import sys
from typing import ClassVar
from pydantic import BaseModel

LOG_LEVEL = 1


class Log(BaseModel):
    """Log to the terminal with colors"""
    # TRACE < DEBUG < INFO < WARN < ERROR < FATAL
    # ClassVar: shared by the class, not fields, used without an instance
    level: ClassVar[int] = LOG_LEVEL

    RED: ClassVar[str] = "\033[0;31m"
    GREEN: ClassVar[str] = "\033[0;32m"
    PURPLE: ClassVar[str] = "\033[0;35m"
    LIGHT_BLUE: ClassVar[str] = "\033[1;34m"
    BOLD: ClassVar[str] = "\033[1m"
    END: ClassVar[str] = "\033[0m"

    @classmethod
    def debug(cls, msg: str | int) -> None:
        """Print debug msg"""
        if cls.level == 0:
            print(cls.PURPLE + str(msg) + cls.END)

    @classmethod
    def info(cls, msg: str) -> None:
        """Print info msg"""
        print(cls.GREEN + msg + cls.END)

    @classmethod
    def error(cls, msg: str) -> None:
        """Print error msg"""
        print(cls.RED + msg + cls.END, file=sys.stderr)

    @classmethod
    def progress(cls) -> None:
        """Show a dot, used to show progress.

        No newline, so it must flush itself using .clear_line
        """
        print(".", end="", flush=True)

    @classmethod
    def clear_line(cls) -> None:
        """Remove dots from the progress bar"""
        # \r goes back to the start of the line, \033[K erases the dots
        print("\r\033[K", end="", flush=True)


if __name__ == "__main__":
    Log.debug("testing debug")
    Log.info("testing info")
    Log.error("testing error")
