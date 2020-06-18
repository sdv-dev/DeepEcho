```
Train:
 - convert data into (L, N, C)
 - convert context into (N, C)
 - feed through model
 - compute loss - requires understanding the channels!

Sample:
 - convert context into (N, C)
 - feed through model
 - convert (L, N, C) into data
```

```json
{
    "metadata": {
        "context_fields": {
            "a": {"type": "float"},
            "b": {"type": "categorical"}
        },
        "sequence_fields": {
            "x": {"type": "float"},
            "y": {"type": "float"},
            "z": {"type": "categorical"}
        }
    },
    "sequences": [
        {
            "context": {
                "a": 0.0,
                "b": "short",
            },
            "data": {
                "x": [0.0, 0.5, 1.0],
                "y": [1.0, 0.5, 1.0],
                "z": ["yes", "no", "yes"]
            }
        },
        {
            "context": {
                "a": 0.0,
                "b": "long",
            },
            "data": {
                "x": [0.0, 0.25, 0.5, 0.75, 1.0],
                "y": [1.0, 0.75, 0.5, 0.25, 1.0],
                "z": ["yes", "no", "no", "no", "yes"]
            }
        }
    ]
}
```
