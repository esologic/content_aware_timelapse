"""
CLI-specific functionality.
"""

from enum import Enum
from typing import Callable, Type, TypeVar

import click
from click import Context, Parameter

# A generic type variable for Enum subclasses, essentially any enum subclass.
E = TypeVar("E", bound=Enum)


def create_enum_option(  # type: ignore[explicit-any]
    arg_flag: str,
    help_message: str,
    default: E,
    input_enum: Type[E],
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """
    Creates a Click option for an Enum type. Resulting input can be given as an index or as the
    string value from the enum.

    :param arg_flag: The argument flag for the Click option (e.g., "--cooler").
    :param help_message: Will be included in the --help message alongside the acceptable inputs
    to the Enum.
    :param default: The default value for the Click option, must be a member of `input_enum`.
    :param input_enum: The Enum class from which the option values are derived.
    :return: A Click option configured for the specified Enum.
    """

    try:
        input_enum(default)
    except ValueError as e:
        raise ValueError("Default value was not a member of the enum!") from e

    options_string = "\n".join(
        [f"   {idx}: {enum_member.value}" for idx, enum_member in enumerate(input_enum)]
    )

    help_string = (
        f"\b\n{help_message}\nOptions below. Either provide index or value:\n{options_string}"
    )

    def callback(_ctx: Context, _param: Parameter, value: str) -> E:
        """
        Validates the input from user.
        :param _ctx: Not used but required by the interface.
        :param _param: Not used but required by the interface.
        :param value: Value from user, should either be an index or an enum string.
        :return: Hydrated input.
        """

        enum_options = list(input_enum)
        try:
            # Try interpreting as an index
            index = int(value)
            if 0 <= index < len(enum_options):
                return enum_options[index]
            else:
                raise click.BadParameter(
                    f"Index out of range. Valid range: 0-{len(enum_options)-1}."
                )
        except ValueError:
            # If not an index, validate as a string
            try:
                return input_enum(value)
            except ValueError as e:
                valid_choices = ", ".join([e.value for e in enum_options])
                raise click.BadParameter(
                    "Invalid choice. "
                    f"Valid names: {valid_choices}, or indices 0-{len(enum_options)-1}."
                ) from e

    return click.option(
        arg_flag,
        type=click.STRING,
        callback=callback,
        help=help_string,
        default=default.value,  # Ensure we use the string value for the default
        show_default=True,
    )
