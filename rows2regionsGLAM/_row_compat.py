from pagerlib.dtypes import Row as _PagerLibRow, ImageSegment


class Row(_PagerLibRow):
    def __init__(self, dict_row=None, **kwargs):
        if dict_row is not None:
            kwargs["segment"] = dict_row.get("segment", None)
            kwargs["children"] = dict_row.get("words", [])
            kwargs["data"] = dict_row.get("data", None)
            if "text" in dict_row and (kwargs["data"] is None or "text" not in kwargs["data"]):
                if kwargs["data"] is None:
                    kwargs["data"] = {}
                kwargs["data"]["text"] = dict_row["text"]
        elif len(kwargs) == 1 and isinstance(list(kwargs.values())[0], dict):
            dict_row = list(kwargs.values())[0]
            kwargs.clear()
            kwargs["segment"] = dict_row.get("segment", None)
            kwargs["children"] = dict_row.get("words", [])
            kwargs["data"] = dict_row.get("data", None)
        super().__init__(**kwargs)
