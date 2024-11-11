import os


def parse_slurmise_record_args(args: list[str]) -> dict:
    """
    Parse the arguments following the `slurmise record` command.

    Expects to receive a list of strings from the `ctx.args` object in the click context. This will have click recognized arguments removed.

    We only allow options to have one value, if multiple values are needed, they need to be quoted.

    For example, the following `slurmise record` command would be parsed:
    `cmd subcmd -o -k 2 -j -i 3 -m fast -q=5 file.json` into the following dictionary:

    :example:

                {
                    | 'cmd': ['cmd', 'subcmd', '-o', '-k', '2', '-j', '-i', '3', '-m', 'fast', '-q=5' ,'file.json'],
                    | 'positional': ['cmd', 'subcmd', 'file.json'],
                    | 'options': {'-k': '2', '-i': '3', '-m': 'fast', '-q': '5'},
                    | 'flags': {'-o': True, '-j': True}}

    """

    parsed_args = {
        "cmd": args,
        "positional": [],
        "options": {},
        "flags": {},
    }

    # Handle the flags and options
    prev_flag = None
    for i, arg in enumerate(args):
        if not arg.startswith("-"):
            if prev_flag:
                parsed_args["options"][prev_flag] = arg
                if prev_flag in parsed_args["flags"]:
                    del parsed_args["flags"][prev_flag]
                prev_flag = None
            else:
                parsed_args["positional"].append(arg)

        elif "=" in arg:
            flag, value = arg.split("=")  # NOTE assumes only 1 equals sign
            parsed_args["options"][flag] = value
        else:
            parsed_args["flags"][arg] = True
            prev_flag = arg

    return parsed_args


def process_slurmise_record_args(parsed_args: dict) -> dict:
    """
    Process the arguments following the `slurmise record` command.

    - Identify file paths and record metadata about the files
    - Convert argument values to appropriate types based on user configuration or type inference

    cmd
    `slurmise record --schema --name myjob --numerics n:3,q:17.4 --categories c:cat1,cat2 --flags --verbose`

    For example, the following parsed_args:
    {
        "cmd": ["sort", "-k", "2" "tests/sacct_output.json"],
        "positional": ["sort", "tests/sacct_output.json"],
        "options": {"-k": "2"},
        "flags": {},
    },

    would be processed into:
    [
        {
            "name": "sort",
            "arg_type": "positional",
            "type": "string",
            "value": "sort",
        },
        {
            "name": "tests/sacct_output.json",
            "arg_type": "positional",
            "type": "file",
            "size": 13398,
        },
        {"name": "-k", "arg_type": "option", "type": "string", "value": "-k"},
    ]
    """

    def process_value(value: str) -> dict:
        if os.path.exists(value):
            file_size = os.path.getsize(value)
            return {"type": "file", "size": file_size}

        # Try to convert to a numeric type (int, then float, then str)
        try:
            return {"type": "numeric", "value": int(value)}
        except ValueError:
            try:
                return {"type": "numeric", "value": float(value)}
            except ValueError:
                return {"type": "string", "value": value}

    # Loop through any positional arguments
    positional = [
        {"name": arg, "arg_type": "positional", **process_value(arg)}
        for arg in parsed_args["positional"]
    ]
    options = [
        {"name": arg, "arg_type": "option", **process_value(arg)}
        for arg in parsed_args["options"]
    ]
    flags = [
        {"name": arg, "arg_type": "flag", "value": True} for arg in parsed_args["flags"]
    ]

    return positional + options + flags
