import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(filename)s - line %(lineno)d - %(levelname)s - %(message)s'
)

# Optionally get a logger object
logger = logging.getLogger(__name__)