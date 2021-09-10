from ggce.utils.combinatorics import ConfigurationSpaceGenerator
from ggce.utils.logger import Logger


# TODO: make this parallel, allow the user to pass a communicator, etc.
def precompute_basis(
    absolute_extent, M, N, max_boson_per_site,
    default_console_logging_level='INFO', log_file=None
):
    """Summary
    
    Parameters
    ----------
    absolute_extent : TYPE
        Description
    M : TYPE
        Description
    N : TYPE
        Description
    max_boson_per_site : TYPE
        Description
    default_console_logging_level : str, optional
        Description
    log_file : None, optional
        Description
    """

    logger = Logger(log_file)
    logger.adjust_logging_level(default_console_logging_level)
    csg = ConfigurationSpaceGenerator(
        absolute_extent, M, N, max_boson_per_site
    )
