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
        "question": "Is the horizontal sliding door, with a grey handle, in the back of the table:",
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
            "shape": "rectangular cube",
            "material": "Not specified, but likely plastic",
            "position": "Located on the workbench surface in the top left area",
            "extra": "The cube has a jagged bottom",
        },
        "missing_property": "color",
        "classes": ["green ", "white"],
        "states": ["on", "off"],
    },
    "slider": {
        "properties": {
            "shape": "rectangular door",
            "material": "wood",
            "color": "brown",
            "position": "Located at the back of the workbench and to the left a grey lever.",
            "extra": "The sliding door has a grey handle to move it left or right. If it is moved to the left, then the right side of the cabinet is open. If it is moved to the right, then the left side of the cabinet is open. Don't mix it up with the lever or drawer. Each of these have grey handles aswell.",
        },
        "missing_property": "state",
        "classes": ["moved to the left", "moved to the right"],
        "states": ["left", "right"],
    },
    "red_block": {
        "properties": {
            "shape": "rectangular cube",
            "material": "Not specified, but likely plastic",
            "color": "red",
            "extra": "Can be located on the workbench surface, in the drawer, or not visible anywhere in the scene. Also there can be pink and blue blocks in the same locations, so don't mix them up.",
        },
        "missing_property": "position",
        "classes": [
            "visible and located on the surface of the workbench",
            "visible and located inside the drawer infront of the workbench",
            "not visible anywhere in the scene",
        ],
        "states": ["table", "drawer", "mia"],
    },
    "pink_block": {
        "properties": {
            "shape": "rectangular cube",
            "material": "Not specified, but likely plastic",
            "color": "pink",
            "extra": "Can be located on the workbench surface, in the drawer, or not visible anywhere in the scene. Also there can be red and blue blocks in the same locations, so don't mix them up.",
        },
        "missing_property": "position",
        "classes": [
            "visible and located on the surface of the workbench",
            "visible and located inside the drawer infront of the workbench",
            "not visible anywhere in the scene",
        ],
        "states": ["table", "drawer", "mia"],
    },
    "blue_block": {
        "properties": {
            "shape": "rectangular cube",
            "material": "Not specified, but likely plastic",
            "color": "blue",
            "extra": "Can be located on the workbench surface, in the drawer, or not visible anywhere in the scene. Also there can be red and pink blocks in the same locations, so don't mix them up.",
        },
        "missing_property": "position",
        "classes": [
            "visible and located on the surface of the workbench",
            "visible and located inside the drawer infront of the workbench",
            "not visible anywhere in the scene",
        ],
        "states": ["table", "drawer", "mia"],
    },
}
