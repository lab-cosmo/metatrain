import difflib

import jsonschema
from jsonschema.exceptions import ValidationError


def validate(instance, schema, cls=None, *args, **kwargs) -> None:
    """Validate an instance under the given schema.

    Function similar to :py:class:`jsonschema.validate` but displaying only the human
    readable error message without showing the reference schema and path if the instance
    is invalid. In addition, if the error is caused by unallowed
    ``additionalProperties`` the closest matching properties will be suggested.

    :param instance: Instance to validate
    :param schema: Schema to validate with
    :raises jsonschema.exceptions.ValidationError: If the instance is invalid
    :raises jsonschema.exceptions.SchemaError: If the schema itself is invalid
    """
    try:
        jsonschema.validate(instance, schema, cls=cls, *args, **kwargs)  # noqa: B026
    except ValidationError as error:
        if error.validator == "additionalProperties":

            known_properties = error.schema["properties"].keys()
            unknown_properties = error.instance.keys() - known_properties

            closest_matches = []
            for name in unknown_properties:
                closest_match = difflib.get_close_matches(
                    word=name, possibilities=known_properties
                )

                if closest_match:
                    closest_matches.append(f"'{closest_match[0]}'")

            if closest_matches:
                error.message += f". Do you mean {', '.join(closest_matches)}?"

        raise ValidationError(error.message)
