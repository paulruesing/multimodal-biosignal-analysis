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
    if len(input_str) < line_break_every:
        return input_str

    output_str = ""
    last_break = 0

    for break_ind in range(0, len(input_str), line_break_every):
        end_break = min(break_ind + line_break_every, len(input_str))

        # Search for next whitespace within the allowed excess range
        for increment in range(max_excess_letters):
            search_pos = break_ind + line_break_every + increment
            if search_pos >= len(input_str):
                break
            if input_str[search_pos] == " ":
                end_break = search_pos
                break

        # Add line up to end_break (excluding the space at end_break)
        output_str += input_str[last_break:end_break].strip() + "\n"

        # Move past the space (if there is one at end_break)
        last_break = end_break + 1 if end_break < len(input_str) and input_str[end_break] == " " else end_break

    # Add remaining text if any
    if last_break < len(input_str):
        output_str += input_str[last_break:].strip()

    return output_str
