TASKS_V1 = {
    "drawer": {
        "question": "Is the wooden, brown drawer under the wooden, brown table:",
        "classes": ["open", "closed"],
        "states": ["open", "closed"],
    },
    "light": {
        "question": "Is the rectangular shaped light on the top left:",
        "classes": ["green ", "white"],
        "states": ["on", "off"],
    },
    "slider": {
        "question": "Is the horizontal slider door, with a grey handle, in the back of the table:",
        "classes": ["moved to the left", "moved to the right"],
        "states": ["left", "right"],
    },
    "red_block": {
        "question": "Is the red block:",
        "classes": [
            "visible and on the brown, wooden table",
            "visible and in the brown, wooden drawer",
            "not visible anywhere",
        ],
        "states": ["table", "drawer", "mia"],
    },
    "pink_block": {
        "question": "Is the pink block:",
        "classes": [
            "visible and on the brown, wooden table",
            "visible and in the brown, wooden drawer",
            "not visible anywhere",
        ],
        "states": ["table", "drawer", "mia"],
    },
    "blue_block": {
        "question": "Is the blue block:",
        "classes": [
            "visible and on the brown, wooden table",
            "visible and in the brown, wooden drawer",
            "not visible anywhere",
        ],
        "states": ["table", "drawer", "mia"],
    },
}

TASKS_V2 = {
    "drawer": {
        "properties": {
            "shape": "rectangular box",
            "material": "wood",
            "color": "brown",
            "position": "Located on the front of the workbench",
            "extra": "The drawer has a grey handle to open and close.",
        },
        "missing_property": "state",
        "classes": ["open", "closed"],
        "states": ["open", "closed"],
    },
    "light": {
        "properties": {
            "shape": "cube with uneven bottom",
            "material": "Not specified, but likely plastic",
            "position": "Located on the workbench surface on the top left",
            "extra": "",
        },
        "missing_property": "color",
        "classes": ["green ", "white"],
        "states": ["on", "off"],
    },
    "slider": {
        "properties": {
            "material": "wood",
            "color": "brown",
            "position": "Located at the back of the table",
            "extra": "The slider has a grey handle to move it left or right. If it is moved to the left, then the right side of the cabinet is open. If it is moved to the right, then the left side of the cabinet is open.",
        },
        "missing_property": "state",
        "classes": ["moved to the left", "moved to the right"],
        "states": ["left", "right"],
    },
    "red_block": {
        "properties": {
            "material": "Not specified, but likely plastic",
            "color": "red",
            "extra": "Can be located on the table, in the drawer, or not visible anywhere",
        },
        "missing_property": "position",
        "classes": [
            "visible and on the brown, wooden table",
            "visible and in the brown, wooden drawer",
            "not visible anywhere",
        ],
        "states": ["table", "drawer", "mia"],
    },
    "pink_block": {
        "properties": {
            "material": "Not specified, but likely plastic",
            "color": "pink",
            "extra": "Can be located on the table, in the drawer, or not visible anywhere",
        },
        "missing_property": "position",
        "classes": [
            "visible and on the brown, wooden table",
            "visible and in the brown, wooden drawer",
            "not visible anywhere",
        ],
        "states": ["table", "drawer", "mia"],
    },
    "blue_block": {
        "properties": {
            "material": "Not specified, but likely plastic",
            "color": "blue",
            "extra": "Can be located on the table, in the drawer, or not visible anywhere",
        },
        "missing_property": "position",
        "classes": [
            "visible and on the brown, wooden table",
            "visible and in the brown, wooden drawer",
            "not visible anywhere",
        ],
        "states": ["table", "drawer", "mia"],
    },
}
