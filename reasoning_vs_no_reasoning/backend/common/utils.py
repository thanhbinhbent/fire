import json, re
from typing import Any, Union, List, Optional, Dict


def join_segments(*args: Union[str, List[str]], separator: str = '\n\n\n') -> str:
    all_segments = []
    for arg in args:
        if isinstance(arg, list):
            all_segments.extend(arg)
        else:
            all_segments.append(strip_string(str(arg)))
    return strip_string(separator.join(all_segments))


def strip_string(s: str) -> str:
    return s.strip(' \n')


def extract_json_from_output(model_output) -> Optional[Dict]:
    json_match = re.search(r'{.*}', model_output, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError:
            return None
    return None


def to_readable_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)

