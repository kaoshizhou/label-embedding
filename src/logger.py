import logging

class Logger:
    def __init__(self) -> None:
        logging.basicConfig(
            level=logging.DEBUG,
            # filename="./output.log",
            format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        