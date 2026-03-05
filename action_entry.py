from typing import NotRequired, TypedDict

class ActionEntry(TypedDict):
    player: str
    street: str
    action_taken: str
    amount: NotRequired[int]
