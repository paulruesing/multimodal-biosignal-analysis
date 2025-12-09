def str_to_float(input: str, is_ger_format=True) -> float:
    """ Converts string to float, ignoring units and eventually converting from GER format (with comma as decimal point and dot magnitude indicator). """
    # convert format:
    if is_ger_format:
        input = input.replace('.', '').replace(',', '.')

    # drop letters:
    input = ''.join([char for char in input if not char.isalpha()])

    # convert to float and return:
    if input == '': return 0.0
    else:
        return float(input.strip())


def enter_line_breaks(input_str: str, line_break_every: int = 110, max_excess_letters: int = 15) -> str:
    if len(input_str) < line_break_every: return input_str
    output_str = ""
    last_break = 0
    for break_ind in range(0, len(input_str), line_break_every):
        for increment in range(max_excess_letters):  # find next whitespace
            try:
                if input_str[break_ind + line_break_every + increment] == " ":
                    end_break = break_ind + line_break_every + increment
            except IndexError: end_break = break_ind + line_break_every
        output_str += input_str[last_break:end_break+1] + "\n"
        last_break = end_break + 1
    return output_str