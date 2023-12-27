import logging


def remove_all_handlers():
    logger = logging.getLogger()
    logger.handlers.clear()


def set_loggers(path_log=None, logging_level=0, b_stream=False, b_debug=False, del_prev_handler=False):

    # std. logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    if path_log:
        if del_prev_handler:
            logger.removeHandler(logger.handlers[-1])
        file_handler = logging.FileHandler(path_log)
        logger.addHandler(file_handler)

    # plot to console
    if b_stream:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)
