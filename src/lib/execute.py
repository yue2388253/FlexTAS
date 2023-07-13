import argparse
import inspect


def execute_from_command_line(func):
    """
    Executes a given function using arguments provided from the command line.

    This function uses the inspect module to determine the required and optional
    arguments of the `func` function and their annotations. It then creates an
    ArgumentParser object and adds the function's arguments to it.

    The command-line arguments are expected to be provided in the format
    `--arg_name arg_value`. The function arguments can be either required or
    optional. If an optional argument is not provided, its default value from
    the function definition is used.

    After parsing the command-line arguments, this function calls `func` with
    the parsed arguments.

    Parameters
    ----------
    func : Callable
        The function to be executed. This function can have any number of
        required or optional arguments.

    Raises
    ------
    argparse.ArgumentError
        If a required argument is not provided in the command line.
    """
    # Get the signature of the function
    sig = inspect.signature(func)

    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    for name, param in sig.parameters.items():
        arg_type = param.annotation if param.annotation is not param.empty else str
        if param.default is param.empty:  # it's a required argument
            parser.add_argument('--' + name, required=True, type=arg_type)
        else:  # it's an optional argument, use default value from function definition
            parser.add_argument('--' + name, default=param.default, type=arg_type)

    args = parser.parse_args()

    # Convert args to a dictionary
    args_dict = vars(args)

    # Call the function with the arguments
    func(**args_dict)
