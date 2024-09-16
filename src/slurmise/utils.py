import json

def parse_slurmise_record_args(args: list[str]) -> dict:
    """Parse the arguments following the `slurmise record` command."""
    #NOTE positional arguments must be listed before first flag
    #NOTE we only allow options to have one value
    #NOTE for example the following `slurmise record` command would be parsed:
    #NOTE `slurmise record cmd subcmd -o -k 2 -j -i 3 -m fast -q=5`
    #NOTE into the following dictionary:
    #NOTE {
    #NOTE     'cmd': ['cmd', 'subcmd', '-o', '-k', '2', '-j', '-i', '3', '-m', 'fast', '-q=5'],
    #NOTE     'positional': ['cmd', 'subcmd'],
    #NOTE     'options': {'-k': '2', '-i': '3', '-m': 'fast'},
    #NOTE     'flags': {'-o': True, '-v': True, '-j': True}
    #NOTE }
    parsed_args = {
        'cmd': args,
        'positional': [],
        'options': {},
        'flags': {},
    }

    #Handle the positional arguments first
    for i, arg in enumerate(args):
        if arg.startswith('-'):
            break
        parsed_args['positional'].append(arg)

    args = args[i:]

    #Handle the flags and options
    prev_flag = None
    breakpoint()
    print(args)
    for i,arg in enumerate(args):
        if not arg.startswith('-'):
            parsed_args['options'][args[i-1]] = arg
            if arg in parsed_args['flags']:
                del parsed_args['flags'][arg]
        else:
            if '=' in arg:
                flag, value = arg.split('=') #NOTE assumes only 1 equals sign
                parsed_args['options'][flag] = value
            else:
                parsed_args['flags'][arg] = True


    return parsed_args    